# Polypseg

Implementation of **MT-SAMPolyp**, a semi-supervised polyp segmentation framework that integrates the MedSAM foundation model into a mean-teacher scheme to improve the reliability of pseudo-labels. The student network combines an EfficientNet-B1 encoder with Mamba layers to capture both local and global context, and is trained with a boundary-aware consistency loss at the feature level.

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
