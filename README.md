# Polypseg

Official implementation of **MT-SAMPolyp**, a framework for semi-supervised polyp segmentation presented in *"MedSAM-guided Boundary-aware Consistency Learning with CNN-Mamba Architecture for Semi-Supervised Polyp Segmentation"*.

Existing semi-supervised approaches based on consistency regularization and pseudo-labeling are often limited by the low reliability of pseudo-labels produced from unlabeled images, which introduces noise into training and weakens the consistency assumption. MT-SAMPolyp addresses this problem by incorporating the MedSAM foundation model into a mean-teacher framework. The teacher, updated as an exponential moving average of the student, produces predictions on unlabeled images that are used to construct mixed samples and box prompts, from which MedSAM-Lite generates refined pseudo-labels to supervise the student.

The student network, PCRN, follows a hybrid CNN-Mamba design. An EfficientNet-B1 backbone extracts local texture features, while Mamba-based layers model long-range spatial dependencies and global context. The decoder employs boundary feature enhancement blocks (BFEB), and a boundary-aware consistency loss is applied at the feature level to obtain sharper edges and more accurate delineation of small or irregular polyps. The method is evaluated on five public datasets: Kvasir-SEG, CVC-ClinicDB, ETIS, CVC-ColonDB and CVC-300.

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
