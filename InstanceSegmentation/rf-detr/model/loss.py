import torch


def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    符合PyTorch风格的Dice Loss实现（精简版）
    适配多类别语义分割，兼容批量计算
    """
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth  # 避免分母为0
        self.reduction = reduction  # 结果聚合方式：none/mean/sum

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, H, W) → 未经过softmax，先转概率
        # targets: (B, H, W) → 像素级类别ID
        inputs = F.softmax(inputs, dim=1)  # 类别维度归一化
        
        # 展平维度：(B, C, H*W) 和 (B, H*W)
        B, C = inputs.shape[:2]
        inputs = inputs.reshape(B, C, -1)
        targets = targets.reshape(B, -1)
        
        # 对每个类别生成二值掩码：(B, C, H*W)
        targets_onehot = F.one_hot(targets, num_classes=C).permute(0, 2, 1).float()
        
        # 计算交集和并集（批量维度求和）
        intersection = (inputs * targets_onehot).sum(dim=-1)  # (B, C)
        union = inputs.sum(dim=-1) + targets_onehot.sum(dim=-1)  # (B, C)
        
        # 计算Dice系数和Loss
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1. - dice  # (B, C)
        
        # 结果聚合
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss

# 测试使用
if __name__ == "__main__":
    loss_fn = DiceLoss()
    # 模拟输入：B=2, C=3, H=4, W=4
    inputs = torch.randn(2, 3, 4, 4)
    targets = torch.randint(0, 3, (2, 4, 4))
    loss: torch.Tensor = loss_fn(inputs, targets) 
    print(f"Dice Loss值: {loss.item():.4f}")  # 输出示例：0.6523