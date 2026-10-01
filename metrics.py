import numpy as np
import torch


def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = pred.round().float()
    pred_binary_inverse = (pred_binary == 0).float()
    gt_binary = gt.round().float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        TP = torch.tensor(1.0, device=pred.device)

    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1 = 2 * precision * recall / (precision + recall)
    f2 = 5 * precision * recall / (4 * precision + recall)
    acc = (TP + TN) / (TP + FP + FN + TN)
    iou_poly = TP / (TP + FP + FN)
    iou_bg = TN / (TN + FP + FN)
    iou_mean = (iou_poly + iou_bg) / 2.0

    size = pred.size(0)
    pred_flat = pred.view(size, -1)
    target_flat = gt.view(size, -1)
    intersection = pred_flat * target_flat
    dice = torch.mean((2 * intersection.sum(1) + 1e-8) / (pred_flat.sum(1) + target_flat.sum(1) + 1e-8))

    return {
        'recall': recall, 'specificity': specificity, 'precision': precision, 'F1': f1, 'Dice': dice, 'F2': f2,
        'ACC_overall': acc, 'IoU_poly': iou_poly, 'IoU_bg': iou_bg, 'IoU_mean': iou_mean,
    }


class Metrics:
    def __init__(self, metrics_list):
        self.metrics = {m: [] for m in metrics_list}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.metrics:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.metrics[k].append(v)

    def mean(self):
        return {k: np.mean(v) for k, v in self.metrics.items()}

    def clean(self):
        for k in self.metrics:
            self.metrics[k].clear()


@torch.no_grad()
def valid(model, loader, device):
    model.eval()
    metrics = Metrics(['recall', 'specificity', 'precision', 'Dice', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])
    for image, mask in loader:
        output = model(image.to(device))
        metrics.update(**evaluate(output, mask.to(device)))
    return metrics.mean()
