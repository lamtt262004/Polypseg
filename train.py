import argparse
import os

import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import PolypDS, train_transform, val_transform
from losses import BceDiceLoss
from metrics import valid
from models import PCRN

TEST_SETS = {
    'Kvasir': 'test_kvasir',
    'ETIS': 'test_etis',
    'CVC-300': 'test_cvc300',
    'CVC-ClinicDB': 'test_clinic',
    'CVC-ColonDB': 'test_colon',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_dir', default='checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = PolypDS(args.data, 'train', train_transform)
    val_ds = PolypDS(args.data, 'val', val_transform)

    n_labeled = max(1, int(round(len(train_ds) * args.ratio)))
    indices = np.random.RandomState(1000 + args.seed).permutation(len(train_ds))[:n_labeled].tolist()
    print(f'Labeled: {n_labeled}/{len(train_ds)}')

    train_loader = DataLoader(Subset(train_ds, indices), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=args.num_workers)

    model = PCRN().to(device)
    criterion = BceDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6)

    ckpt_path = os.path.join(args.save_dir, f'PCRN_{int(args.ratio * 100)}pct_seed{args.seed}.pth')
    best_dice = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for images, gts in pbar:
            images, gts = images.to(device), gts.to(device)
            optimizer.zero_grad()
            pred, pred_side = model(images)
            loss = criterion(pred, gts) + criterion(pred_side, gts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix_str(f'loss: {loss.item():.4f}')

        res = valid(model, val_loader, device)
        scheduler.step(res['Dice'])
        print(f'Epoch {epoch:3d} | loss {total_loss / len(train_loader):.4f} '
              f'| val Dice {res["Dice"]:.4f} | val IoU {res["IoU_mean"]:.4f}')

        if res['Dice'] > best_dice:
            best_dice = res['Dice']
            torch.save(model.state_dict(), ckpt_path)

    print(f'Best val Dice: {best_dice:.4f}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    run_tests(model, args.data, device)


def run_tests(model, data_path, device):
    for name, split in TEST_SETS.items():
        loader = DataLoader(PolypDS(data_path, split, val_transform), batch_size=1, shuffle=False)
        r = valid(model, loader, device)
        print(f'{name:<13} Dice {r["Dice"] * 100:.2f} | IoU {r["IoU_poly"] * 100:.2f} '
              f'| Acc {r["ACC_overall"] * 100:.2f} | Recall {r["recall"] * 100:.2f}')


if __name__ == '__main__':
    main()
