# MSSA-Net: Mamba-Driven Spatial-Frequency Synergy for Medical Image Segmentation**
[[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]][![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1%2Bcu117-orange)] [[![CUDA](https://img.shields.io/badge/CUDA-11.6%2B-red)]]

***Abstract***: Precise segmentation of medical images (abdominal CT, cardiac MRI) is critical for clinical diagnosis. Convolutional neural networks (CNNs) struggle with long-range dependencies, while traditional Transformers suffer from high computational costs. To address these issues, we propose **MSSA-Net**, a novel architecture characterized by three key innovations:
**(i)** It introduces a **Spatial-Frequency Fusion Block (SMFB)** at the encoder stage, which combines Mamba's SS2D module (for efficient long-range modeling) and a Dual-Domain Enhancement Module (DDEM) (for spatial detail and frequency semantic reinforcement). This not only expands the receptive field but also preserves fine-grained boundary information.
**(ii)** A **Hierarchical Semantic Link (HSL)** module is integrated into the bottleneck layer, consisting of a Reconstruction of Global-Local Features (RUG) unit and a Cross-Source Alignment Attention (CAA) module. This harmonizes cross-scale semantic information and avoids feature drift.
**(iii)** The framework achieves linear computational complexity (O(n log n)) via Mamba's selective scanning mechanism, outperforming 14 state-of-the-art methods on the Synapse and ACDC datasets.

>** Key Reminder**: Mamba module is a mandatory dependency — code will fail to run if Mamba is not properly configured.


***1. Dependencies and Installation (Full Details)***
**1.1 Clone this repo:**
```bash
git clone https://github.com/你的仓库地址/MSSA-Net.git](https://github.com/yjy2001230/MSSA-Net.
cd MSSA-Net
1.2 Create conda environment and install dependencies (Exact Versions):
# Step 1: Create and activate environment (Python 3.8 is mandatory)
conda create -n mssanet python=3.8 -y
conda activate mssanet

# Step 2: Install PyTorch (match CUDA version strictly)
# For CUDA 11.7 (recommended)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# For CUDA 11.6 (alternative)
# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Step 3: Install core medical imaging dependencies (fixed versions to avoid conflicts)
pip install monai==1.1.0 numpy==1.23.5 scipy==1.10.1 scikit-image==0.20.0 tqdm==4.64.1
pip install opencv-python==4.7.0.72 tensorboard==2.11.2 scikit-learn==1.2.2 matplotlib==3.7.1
pip install thop==0.1.1.post2209072238 h5py==3.8.0 SimpleITK==2.2.1 medpy==0.4.0 yacs==0.1.8

# Step 4: Install Mamba core dependencies (REQUIRED, no version substitution)
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1

# If Mamba installation fails (common fixes):
# Fix 1: Build from source (bypass PyPI restrictions)
# pip install --no-build-isolation mamba_ssm
# Fix 2: Update setuptools first
# pip install --upgrade setuptools wheel
# pip install mamba_ssm==1.0.1

1.3 Verify environment (Critical Step to Ensure Runability):
Run this script to confirm all dependencies (especially Mamba) are functional:
import torch
import monai
import mamba_ssm
from mamba_ssm import Mamba

# Check core library versions
print(f"✅ PyTorch version: {torch.__version__} (must be 1.13.1+cu117/cu116)")
print(f"✅ MONAI version: {monai.__version__} (must be 1.1.0)")
print(f"✅ Mamba-SSM version: {mamba_ssm.__version__} (must be 1.0.1)")

# Check CUDA availability (Mamba requires GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise RuntimeError("❌ Mamba requires CUDA GPU — code cannot run on CPU!")
print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")

# Test Mamba module initialization (core check)
try:
    mamba_block = Mamba(
        d_model=256,  # Exact value matching model config
        d_state=16,    # Critical for model compatibility
        d_conv=4,      # Fixed parameter in MSSA-Net
        expand=2       # Do not modify
    ).to(device)
    # Test forward pass (ensure no runtime errors)
    test_input = torch.randn(2, 256, 224, 224).to(device)  # Batch=2, Channels=256, H/W=224
    test_output = mamba_block(test_input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    print(f"✅ Mamba forward pass successful — output shape: {test_output.shape}")
    print("✅ All dependencies are configured correctly!")
except Exception as e:
    print(f"❌ Environment error (fix before running): {str(e)}")
    print("🔧 Common fixes:")
    print("   1. Ensure CUDA 11.6+/PyTorch 1.13.1+ are installed")
    print("   2. Reinstall Mamba via source: pip install --no-build-isolation mamba_ssm")
    print("   3. Check GPU memory (Mamba requires ≥8GB VRAM)")
2. Dataset Preparation (Exact Structure for Runability)
2.1 Download datasets (Official Sources Only):
- The Synapse dataset we used are provided by TransUnet's authors. [![](https://img.shields.io/badge/Dataset-🚀Synapse-blue.svg)](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License). The ACDC dataset can be obtained from [![](https://img.shields.io/badge/Dataset-🚀ACDC-blue.svg)](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
- I am not the owner of these two preprocessed datasets. **Please follow the instructions and regulations set by the official releaser of these two datasets.** The directory structure of the whole project is as follows:
```
2.2 Mandatory Dataset Structure (Code Hardcodes This Path):
data/
├── Synapse/
│   ├── raw/               
│   │   ├── train/img/      
│   │   ├── train/label/   
│   │   ├── test/img/      
│   │   └── test/label/     
│   └── processed/          
│       ├── train/
│       └── test/
└── ACDC/
    ├── raw/
    │   ├── train/img/     
    │   ├── train/label/    
    │   ├── test/img/      
    │   └── test/label/    
    └── processed/
        ├── train/
        └── test/
