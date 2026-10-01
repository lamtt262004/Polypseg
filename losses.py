import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.bce_loss = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, pred, target):
        size = pred.size(0)
        return self.bce_loss(pred.view(size, -1), target.view(size, -1))


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        return 1 - dice_score.sum() / size


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.bce = BCELoss(weight, reduction=reduction)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)


def calculate_distance(strong, weak):
    distance = 0
    for s, w in zip(strong, weak):
        B = s.shape[0]
        s_vec = torch.logit(s.clamp(1e-6, 1 - 1e-6)).view(B, -1)
        w_vec = torch.logit(w.clamp(1e-6, 1 - 1e-6)).view(B, -1).detach()
        distance += 1 - F.cosine_similarity(s_vec, w_vec, dim=1).mean()
    return distance / len(strong)


def compute_attention_mask_boundary(binary_mask):
    dist_fg = distance_transform_edt(binary_mask == 0)
    dist_bg = distance_transform_edt(binary_mask == 1)
    return torch.tensor(1 / (1 + dist_fg + dist_bg), dtype=torch.float32)


def guide_fusion_loss(strong, weak):
    dice = DiceLoss()(strong, weak)
    cos_loss = calculate_distance(strong, weak)
    attention_mask = compute_attention_mask_boundary((weak > 0.5).float().cpu().numpy()).to(weak.device)
    return (cos_loss + dice * attention_mask).mean()
