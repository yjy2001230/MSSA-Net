# MSSA-Net

PyTorch implementation of MSSA-Net for medical image segmentation.

This repository contains training and evaluation code for two public benchmarks:
Synapse (multi-organ CT) and ACDC (cardiac MRI). The scripts follow a slice-wise
2D training/evaluation protocol (224×224).

## 1. Requirements

- Python 3.8
- CUDA GPU recommended for training
- All required Python packages are listed in `requirements.txt`.

## 2. Installation

```bash
# 1) create env
conda create -n mssa python=3.8 -y
conda activate mssa

# 2) install deps
pip install -r requirements.txt
```

### Optional: Mamba-related dependencies
If you enabled Mamba/SSM components in your code and they are not covered by your environment,
install the corresponding packages for your CUDA/PyTorch build (typical examples: `mamba-ssm`,
`causal-conv1d`). If installation fails, it is usually a CUDA/PyTorch build mismatch.

## 3. Data preparation

### 3.1 Download
- Synapse (MICCAI 2015 Multi-Atlas Abdomen Labeling): download from Synapse portal (login required).
- ACDC (Automated Cardiac Diagnosis Challenge): download from the official ACDC website.

### 3.2 Directory layout

Place the preprocessed data with the following structure:

```text
.
├── data
│   ├── Synapse
│   │   ├── train_npz
│   │   │   ├── case0005_slice000.npz
│   │   │   └── *.npz
│   │   └── test_vol_h5
│   │       ├── case0001.npy.h5
│   │       └── *.npy.h5
│   └── ACDC
│       ├── train
│       │   ├── case_001_sliceED_0.npz
│       │   └── *.npz
│       └── test
│           ├── case_002_volume_ED.npz
│           └── *.npz
└── lists
    └── (train/val/test split files if required by your scripts)
```

If your scripts use list files, put split files under `./lists` and pass `--list_dir ./lists`.

## 4. Reproducibility (paper settings)

Typical settings used in the paper:
- input size: 224
- optimizer: SGD
- base learning rate: 0.05
- momentum: 0.9
- weight decay: 1e-4
- LR schedule: polynomial decay
- loss: weighted CE + Dice
- Synapse: 600 epochs, batch size 12
- ACDC: 1000 epochs, batch size 12

You can inspect all configurable arguments via:

```bash
python train.py -h
python test.py -h
```

## 5. Training

### 5.1 Synapse
```bash
python train.py --dataset Synapse --output_dir ./model_output_Synapse --max_epochs 600 --batch_size 12
```

### 5.2 ACDC
```bash
python train.py --dataset ACDC --output_dir ./model_output_ACDC --max_epochs 1000 --batch_size 12
```

> Note: If your code requires an explicit dataset root flag (e.g., `--root_path`), use the exact flag name defined in your argument parser.

## 6. Testing / Inference

### 6.1 Synapse
```bash
python test.py --dataset Synapse --is_saveni True --output_dir ./model_output_Synapse --max_epoch 600 --batch_size 12 --test_save_dir ./model_output_Synapse/predictions
```

### 6.2 ACDC
```bash
python test.py --dataset ACDC --is_saveni True --output_dir ./model_output_ACDC --max_epoch 1000 --batch_size 12 --test_save_dir ./model_output_ACDC/predictions
```

## 7. Outputs

- Checkpoints and logs are written to `--output_dir`.
- If `--is_saveni True` is enabled, predicted masks/volumes are saved to `--test_save_dir`.

## 8. Troubleshooting

- `ModuleNotFoundError`: re-check environment and `pip install -r requirements.txt`.
- CUDA build issues (especially for optional Mamba packages): ensure your PyTorch CUDA version matches your system CUDA toolkit.
