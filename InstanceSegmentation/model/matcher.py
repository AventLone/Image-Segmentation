import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou


class HungarianMatcher:
    def __init__(self, class_cost=1.0, bbox_cost=5.0, giou_cost=2.0):
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def __call__(self, logits, boxes, targets):
        bs, num_queries, num_classes = logits.shape
        indices = []

        probs = logits.softmax(-1)

        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_boxes = targets[b]["boxes"]

            if tgt_boxes.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            # cost
            cost_class = -probs[b][:, tgt_ids]
            cost_bbox = torch.cdist(boxes[b], tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(boxes[b], tgt_boxes)

            C = self.class_cost * cost_class + self.bbox_cost * cost_bbox + self.giou_cost * cost_giou
            C = C.cpu()

            row_ind, col_ind = linear_sum_assignment(C)

            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long)
            ))

        return indices
