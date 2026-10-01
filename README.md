# Polypseg

Semi-supervised polyp segmentation. PCRN (EfficientNet-B1 + Mamba encoder, BFEB decoder) is the student, a mean-teacher EMA model mixes unlabeled images, and MedSAM-Lite refines the pseudo-labels from box prompts.

## Install

```bash
pip install -r requirements.txt
```

`mamba_ssm` needs Linux + CUDA (Colab works).

## Data

A single `.npz` file with keys `train_img`, `train_msk`, `val_img`, `val_msk`, `test_kvasir_*`, `test_etis_*`, `test_cvc300_*`, `test_clinic_*`, `test_colon_*`.

## Run

Quick test on a random image:

```bash
python demo.py
```

Predict one image:

```bash
python demo.py --image sample.jpg --ckpt PCRN.pth --out pred.png
```

Semi-supervised training (72 labeled images, needs `lite_medsam.pth`):

```bash
python train_ssl.py --data polypData.npz --medsam_ckpt lite_medsam.pth --labeled_num 72
```

Supervised baseline (5% labeled):

```bash
python train.py --data polypData.npz --ratio 0.05 --epochs 100
```
