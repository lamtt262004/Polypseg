import torch


def get_bbox256_torch(mask_256, bbox_shift=3):
    B, C, H, W = mask_256.shape
    bboxes256 = torch.ones((B, 1, 4)).to(mask_256.device) * (-100)
    for n in range(B):
        pd_one = mask_256[n, :, :]
        idx_fg = torch.argwhere(pd_one > 0.5)
        if idx_fg.sum() > 0:
            x_min, x_max = torch.min(idx_fg[:, 1]), torch.max(idx_fg[:, 1])
            y_min, y_max = torch.min(idx_fg[:, 0]), torch.max(idx_fg[:, 0])
            x_min = max(0, x_min - bbox_shift)
            x_max = min(W, x_max + bbox_shift)
            y_min = max(0, y_min - bbox_shift)
            y_max = min(H, y_max + bbox_shift)
            bboxes256[n, 0, :] = torch.tensor([x_min, y_min, x_max, y_max])
    return bboxes256
