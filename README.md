# MDSF-Net

PyTorch implementation of MSSA-Net for medical image segmentation.

This repository contains training and evaluation code for two public benchmarks:
Synapse (multi-organ CT) and ACDC (cardiac MRI). The scripts follow a slice-wise
2D training/evaluation protocol (224×224).


## 1. Environment

All Python dependencies are listed in requirements.txt.

Recommended setup:

```bash
conda create -n mdsf python=3.8 -y
conda activate mdsf
pip install -r requirements.txt
```

Notes:
- a CUDA-enabled GPU is recommended for training
- if your requirements include mamba-ssm / causal-conv1d and installation fails, ensure your CUDA version matches the PyTorch build used by your environment


## 2. Datasets

We evaluate on two public benchmarks:
- Synapse multi-organ CT (8 abdominal organs)
- ACDC cardiac MRI (3 cardiac structures)

2.1 Download links (official)
- Synapse (MICCAI 2015 Multi-Atlas Abdomen Labeling):
  https://www.synapse.org/Synapse:syn3193805
  Files page:
  https://www.synapse.org/Synapse:syn3193805/files/
  Note: Synapse requires registration/login to download.
- ACDC (Automated Cardiac Diagnosis Challenge):
  https://www.creatis.insa-lyon.fr/Challenge/acdc/
  Download page:
  https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

2.2 Directory layout

Use the directory layout expected by the training scripts.

A typical layout is:

```text
.
├── datasets
│   └──
├── lists
│   └── 
├── data
│   ├── Synapse
│   │   ├── train_npz
│   │   │   ├── case0005_slice000.npz
│   │   │   └── *.npz
│   │   └── test_vol_h5
│   │       ├── case0001.npy.h5
│   │       └── *.npy.h5
│   └── ACDC
│       ├── train
│       │   ├── case_001_sliceED_0.npz
│       │   └── *.npz
│       ├── test
│       │   ├── case_002_volume_ED.npz
│       │   └── *.npz
│       └── train
│           ├── case_019_sliceED_0.npz
│           └── *.npz
├── networks
│   └── 
├── train
├── test
└── trainer
```

If your scripts use list files:
- put split files under ./lists
- set --list_dir ./lists when running train/test


## 3. Reproducibility settings

Key hyper-parameters (paper setting):
- input size: 224
- optimizer: SGD
- initial learning rate: 0.05
- momentum: 0.9
- weight decay: 1e-4
- lr schedule: polynomial decay
- loss: weighted cross-entropy + dice loss
- training from scratch (no pretrained weights)
- fixed random seed is set in the training script

Training schedule (paper setting):
- Synapse: 1000 epochs
- ACDC: 2000 epochs
- batch size: 6


## 4. Training

Run the following to check available arguments:
```bash
python train.py -h
```

4.1 Train on Synapse

```bash
python train.py --dataset Synapse --output_dir './model_output_Synapse' --max_epochs 600 --batch_size 6
```

4.2 Train on ACDC

```bash
python train.py --dataset ACDC --output_dir './model_output_ACDC' --max_epochs 1000 --batch_size 6
```

If your code requires an explicit dataset root path, add it to the command using the exact flag name in your scripts (for example --root_path or --data_root):

```bash
# example only: use the exact flag defined in your arg parser
python train.py ... --root_path <DATA_ROOT>/Synapse
```


## 5. Testing / Inference

Example:

```bash
python test.py --dataset Synapse --is_saveni True --output_dir './model_output_Synapse' --max_epoch 600 --batch_size 12 --test_save_dir './model_output_Synapse/predictions'
```
```bash
python test.py --dataset ACDC --is_saveni True --output_dir './model_output_ACDC' --max_epoch 1000 --batch_size 12 --test_save_dir './model_output_ACDC/predictions'
```








