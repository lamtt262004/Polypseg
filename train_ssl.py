import argparse
import csv
import os

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PolypDS, TwoStreamBatchSampler, train_transform, val_transform
from losses import BceDiceLoss, guide_fusion_loss
from metrics import valid
from models import PCRN
from models.medsam_lite import load_medsam
from prompts import get_bbox256_torch
from train import run_tests
from utils import generate_model, get_current_consistency_weight, update_ema_variables


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--medsam_ckpt', required=True)
    parser.add_argument('--labeled_num', type=int, default=72)
    parser.add_argument('--labeled_bs', type=int, default=2)
    parser.add_argument('--unlabeled_bs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--thresh', type=float, default=0.7)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--name', default='PCRNet_MedSAM')
    return parser.parse_args()


def mix_unlabeled(ema_model, unlabeled_img, thresh):
    group0, group1 = torch.chunk(unlabeled_img, 2, 0)

    with torch.no_grad():
        ema_gt0, ema_feat0 = ema_model(group0)
        ema_gt1, ema_feat1 = ema_model(group1)

    mix_feat = ema_feat1 / (ema_feat0 + ema_feat1)
    feature_mixed = ema_feat0 * (1.0 - mix_feat) + ema_feat1 * mix_feat

    up0 = F.interpolate(ema_feat0, size=group0.shape[2:], mode='bilinear', align_corners=True)
    up1 = F.interpolate(ema_feat1, size=group0.shape[2:], mode='bilinear', align_corners=True)
    mix_img = up1 / (up1 + up0)
    image_mixed = group0 * (1.0 - mix_img) + group1 * mix_img
    gt_mixed = ema_gt0 * (1.0 - mix_img) + ema_gt1 * mix_img

    mask0 = (ema_gt0 > ema_gt1) * (ema_gt0 > thresh)
    mask1 = (ema_gt1 > ema_gt0) * (ema_gt1 > thresh)
    gt_pseudo = ema_gt0 * mask0 + ema_gt1 * mask1 + gt_mixed * torch.logical_not(torch.logical_or(mask0, mask1))

    return image_mixed, gt_pseudo, feature_mixed


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    lbs = args.labeled_bs

    train_ds = PolypDS(args.data, 'train', train_transform)
    val_ds = PolypDS(args.data, 'val', val_transform)
    sampler = TwoStreamBatchSampler(len(train_ds), args.labeled_num, lbs, args.unlabeled_bs, shuffle=True)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=args.num_workers)

    model = PCRN().to(device)
    ema_model = generate_model(model, ema=True).to(device)
    medsam_model = load_medsam(args.medsam_ckpt).to(device)
    model.train()
    ema_model.eval()
    medsam_model.train()

    criterion = BceDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer_sam = optim.SGD(medsam_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6)

    ckpt_path = os.path.join(args.save_dir, f'{args.name}.pth')
    history_path = os.path.join(args.save_dir, f'{args.name}_history.csv')
    with open(history_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'sam_loss', 'sup_loss', 'con_loss', 'val_dice', 'val_iou'])

    iter_num = 0
    best_dice = 0.0

    for epoch in range(args.epochs):
        model.train()
        sam_loss, sup_loss, cons_loss = 0.0, 0.0, 0.0
        weight = get_current_consistency_weight(epoch + 1, 0.1, 10.0)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for images, gts in pbar:
            images, gts = images.to(device), gts.to(device)
            labeled_img, mask_x = images[:lbs], gts[:lbs]

            image_mixed, gt_pseudo, feature_mixed = mix_unlabeled(ema_model, images[lbs:], args.thresh)

            pred_x_sam, _ = medsam_model(labeled_img, get_bbox256_torch(mask_x == 1))
            pred_u_sam, _ = medsam_model(image_mixed, get_bbox256_torch(gt_pseudo, bbox_shift=10))
            pred_x_sam = torch.sigmoid(pred_x_sam)
            pred_u_sam = torch.sigmoid(pred_u_sam)

            loss_sam = criterion(pred_x_sam, mask_x) + 0.25 * criterion(pred_u_sam, gt_pseudo)
            optimizer_sam.zero_grad()
            loss_sam.backward()
            optimizer_sam.step()

            optimizer.zero_grad()
            pred_gt, pred_feature = model(torch.cat([labeled_img, image_mixed], dim=0))

            loss_sup = criterion(pred_gt[:lbs], mask_x) + criterion(pred_feature[:lbs], mask_x)
            loss_cons_gt = criterion(pred_gt[lbs:], pred_u_sam.detach())
            loss_cons_feat = guide_fusion_loss(pred_feature[lbs:], feature_mixed)
            loss = loss_sup + weight * (loss_cons_gt + loss_cons_feat)
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, 0.99, iter_num)
            iter_num += 1

            sam_loss += loss_sam.item()
            sup_loss += loss_sup.item()
            cons_loss += loss_cons_gt.item() + loss_cons_feat.item()
            pbar.set_postfix_str(f'loss: {loss.item():.4f}')

        n = len(train_loader)
        sam_loss, sup_loss, cons_loss = sam_loss / n, sup_loss / n, cons_loss / n

        res = valid(model, val_loader, device)
        val_dice, val_iou = res['Dice'], res['IoU_mean']
        scheduler.step(val_dice)

        print(f'Epoch {epoch:3d} | sam {sam_loss:.4f} | sup {sup_loss:.4f} | cons {cons_loss:.4f} '
              f'| w {weight:.4f} | val Dice {val_dice:.4f} | val IoU {val_iou:.4f}')

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_path)

        with open(history_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, sam_loss, sup_loss, cons_loss, val_dice, val_iou])

    print(f'Best val Dice: {best_dice:.4f}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    run_tests(model, args.data, device)


if __name__ == '__main__':
    main()
