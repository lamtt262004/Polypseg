import argparse

import cv2
import numpy as np
import torch

from dataset import val_transform
from models import PCRN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None)
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--out', default='pred.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PCRN(pretrained=args.ckpt is None).to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    if args.image:
        img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    else:
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    h, w = img.shape[:2]

    x = torch.from_numpy(val_transform(image=img)['image']).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred, pred_side = model(x)

    print('input:', tuple(x.shape))
    print('output:', tuple(pred.shape), tuple(pred_side.shape))

    mask = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(args.out, mask)
    print('saved:', args.out)


if __name__ == '__main__':
    main()
