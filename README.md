# Polypseg

PCRN for polyp segmentation (EfficientNet-B1 + Mamba encoder, BFEB decoder).

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

Train (5% labeled):

```bash
python train.py --data polypData.npz --ratio 0.05 --epochs 100
```
