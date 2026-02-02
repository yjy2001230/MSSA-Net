# MSSA-Net

Official PyTorch implementation of our paper for medical image segmentation.

This repository provides:
- end-to-end training and evaluation scripts
- dataset preparation convention (Synapse / ACDC)
- reproducible hyper-parameters and command examples
- optional prediction saving for qualitative visualization


## 1. Environment

All Python dependencies are listed in requirements.txt.

Recommended setup:

```bash
conda create -n mssa python=3.8 -y
conda activate mssa
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
<DATA_ROOT>/
  Synapse/
    images/ (or your processed slices)
    labels/
    (other files required by your preprocessing)
  ACDC/
    images/
    labels/
./lists/
  (train/val/test split files for each dataset)
```

If your scripts use list files:
- put split files under ./lists
- set --list_dir ./lists when running train/test

2.3 Data protocol (paper setting)

To keep the evaluation consistent and reproducible:
- 3D volumes are sliced into 2D axial slices
- all slices are resized to 224x224 for both training and testing
- intensity normalization is applied
- data augmentations include random flipping, affine transforms (scale/rotate/translate/shear), and additive Gaussian noise

Synapse split:
- 18 volumes for training, 12 volumes for testing

ACDC split:
- 70 cases for training, 10 for validation, 20 for testing
- training and evaluation are performed at the 2D slice level


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
python train.py   --dataset Synapse   --img_size 224   --max_epoch 1000   --batch_size 6   --base_lr 0.05   --list_dir ./lists   --output_dir ./outputs/synapse
```

4.2 Train on ACDC

```bash
python train.py   --dataset ACDC   --img_size 224   --max_epoch 2000   --batch_size 6   --base_lr 0.05   --list_dir ./lists   --output_dir ./outputs/acdc
```

If your code requires an explicit dataset root path, add it to the command using the exact flag name in your scripts (for example --root_path or --data_root):

```bash
# example only: use the exact flag defined in your arg parser
python train.py ... --root_path <DATA_ROOT>/Synapse
```


## 5. Testing / Inference

Run the following to check available arguments:
```bash
python test.py -h
```

Example:

```bash
python test.py   --dataset Synapse   --img_size 224   --list_dir ./lists   --output_dir ./outputs/synapse   --test_save_dir ./outputs/synapse/predictions
```

Outputs:
- metrics will be printed to stdout (Synapse: mean DSC and mean HD95; ACDC: mean DSC)
- if --test_save_dir is provided, predicted masks will be saved for visual inspection


## 6. Saving predictions (optional)

If you want to save predictions for qualitative comparison:
- set --test_save_dir to a folder path
- the script will dump predicted masks (format depends on the repository implementation)

If you need PNG overlays or NIfTI exports, add a small utility script to convert saved predictions, or use an existing script in the repository if available.


## 7. Troubleshooting

7.1 mamba-ssm / causal-conv1d installation issues
- ensure your CUDA runtime matches the PyTorch version
- in cluster/AutoDL environments, prefer installing wheels that match the CUDA runtime

7.2 Out of memory
- reduce batch_size
- reduce img_size (only if you are not strictly reproducing the paper setting)
- enable mixed precision if your scripts support it

7.3 Determinism
- ensure the seed flag (or deterministic option) is enabled in your script
- full determinism can slightly reduce speed


## 8. Citation

If you use this code, please cite our paper:

```bibtex
@article{mssa_net,
  title={...},
  author={...},
  journal={...},
  year={...}
}
```
