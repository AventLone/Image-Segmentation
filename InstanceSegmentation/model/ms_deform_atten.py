"""
Multi-Scale Deformable Attention Module
"""

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

def _bilinear_grid_sample(
    input: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Bilinear grid sampling compatible with all PyTorch backends including MPS.

    Drop-in replacement for ``F.grid_sample(input, grid, mode='bilinear', ...)``.
    On MPS, ``F.grid_sample`` backward (``grid_sampler_2d_backward``) is not yet
    implemented and silently falls back to CPU.  This function uses gather-based
    index arithmetic — natively supported on every backend — for the MPS path,
    while delegating to ``F.grid_sample`` on CUDA/CPU where its fused kernel is
    faster.  The two paths are numerically identical, so model accuracy is
    unaffected.

    Args:
        input: Feature map of shape ``(N, C, H, W)``.
        grid: Sampling grid of shape ``(N, Hg, Wg, 2)`` with values in ``[-1, 1]``.
        padding_mode: ``"zeros"`` returns 0 for out-of-bounds samples;
            ``"border"`` clamps to the nearest border pixel.
        align_corners: If ``True``, grid extremes ``±1`` map to pixel centres at
            positions ``0`` and ``H-1``/``W-1``.

    Returns:
        Sampled tensor of shape ``(N, C, Hg, Wg)``.
    """
    import torch.nn.functional as F

    if input.device.type != "mps":
        return F.grid_sample(input, grid, mode="bilinear", padding_mode=padding_mode, align_corners=align_corners)

    if padding_mode not in ("zeros", "border"):
        msg = (
            f"Unsupported padding_mode={padding_mode!r} for manual grid sampling. "
            "Only 'zeros' and 'border' are supported in this path."
        )
        raise ValueError(msg)

    N, C, H, W = input.shape
    Hg, Wg = grid.shape[1], grid.shape[2]

    # Unnormalize [-1, 1] → floating-point pixel coordinates
    if align_corners:
        ix = (grid[..., 0] + 1) * (W - 1) / 2  # [N, Hg, Wg]
        iy = (grid[..., 1] + 1) * (H - 1) / 2
    else:
        ix = (grid[..., 0] + 1) * W / 2 - 0.5
        iy = (grid[..., 1] + 1) * H / 2 - 0.5

    ix0 = ix.floor().long()  # top-left corner
    iy0 = iy.floor().long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # Bilinear weights: fractional distance from top-left corner  [N, 1, Hg, Wg]
    # Cast to input.dtype so float16 inputs don't silently upcast to float32.
    wx1 = (ix - ix0.float()).to(input.dtype).unsqueeze(1)
    wy1 = (iy - iy0.float()).to(input.dtype).unsqueeze(1)
    one = wx1.new_tensor(1.0)
    wx0 = one - wx1
    wy0 = one - wy1

    if padding_mode == "border":
        ix0 = ix0.clamp(0, W - 1)
        iy0 = iy0.clamp(0, H - 1)
        ix1 = ix1.clamp(0, W - 1)
        iy1 = iy1.clamp(0, H - 1)
    else:  # zeros: record which corners fall inside the image before clamping
        in_x0 = (ix0 >= 0) & (ix0 < W)  # [N, Hg, Wg]
        in_x1 = (ix1 >= 0) & (ix1 < W)
        in_y0 = (iy0 >= 0) & (iy0 < H)
        in_y1 = (iy1 >= 0) & (iy1 < H)
        ix0 = ix0.clamp(0, W - 1)
        iy0 = iy0.clamp(0, H - 1)
        ix1 = ix1.clamp(0, W - 1)
        iy1 = iy1.clamp(0, H - 1)

    flat = input.flatten(2)  # [N, C, H*W]

    def _gather(iy_: torch.Tensor, ix_: torch.Tensor) -> torch.Tensor:
        idx = (iy_ * W + ix_).flatten(1).unsqueeze(1).expand(N, C, -1)  # [N, C, Hg*Wg]
        return flat.gather(2, idx).view(N, C, Hg, Wg)

    v00 = _gather(iy0, ix0)  # top-left
    v10 = _gather(iy0, ix1)  # top-right
    v01 = _gather(iy1, ix0)  # bottom-left
    v11 = _gather(iy1, ix1)  # bottom-right

    if padding_mode == "zeros":
        v00 = v00 * (in_x0 & in_y0).unsqueeze(1)
        v10 = v10 * (in_x1 & in_y0).unsqueeze(1)
        v01 = v01 * (in_x0 & in_y1).unsqueeze(1)
        v11 = v11 * (in_x1 & in_y1).unsqueeze(1)

    return wx0 * wy0 * v00 + wx1 * wy0 * v10 + wx0 * wy1 * v01 + wx1 * wy1 * v11


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """ "for debug and test only, need to use cuda version instead"""
    # B, n_heads, head_dim, N
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H, W) in enumerate(value_spatial_shapes):
        # B, n_heads, head_dim, H, W
        value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
        # B, Len_q, n_heads, P, 2 -> B, n_heads, Len_q, P, 2 -> B*n_heads, Len_q, P, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # B*n_heads, head_dim, Len_q, P
        sampling_value_l_ = _bilinear_grid_sample(value_l_, sampling_grid_l_, padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (B, Len_q, n_heads, L * P) -> (B, n_heads, Len_q, L, P) -> (B*n_heads, 1, Len_q, L*P)
    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Len_q, L * P)
    # B*n_heads, head_dim, Len_q, L*P
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()



def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module"""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the"
                " dimension of each attention head a power of 2"
                " which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

        self._export = False

    def export(self):
        """export mode"""
        self._export = True

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        r"""
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1],
                                           top-left (0,0), bottom-right (1, 1), including padding area
                                           or (N, Length_{query}, n_levels, 4), add additional (w, h)
                                           to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1,
                                           H_0*W_0+H_1*W_1+H_2*W_2, ...,
                                           H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding
                                           elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )
        attention_weights = F.softmax(attention_weights, -1)

        value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output
