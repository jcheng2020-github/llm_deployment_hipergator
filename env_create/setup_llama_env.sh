#!/bin/bash
#SBATCH --job-name=setup_llama_env
#SBATCH --time=8:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=setup_%j.out

echo "===== START JOB ====="
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM job id: $SLURM_JOB_ID"

################################################################################
# 1. Load CUDA
################################################################################
module load cuda/12.8.1
echo "Loaded CUDA version:"
nvcc --version || echo "nvcc not available (wheel CUDA will be used)."

################################################################################
# 2. Load conda
################################################################################
# If your cluster uses a module for conda, load it. Example:
module load conda

# Initialize shell
source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"

################################################################################
# 3. Create conda environment
################################################################################
ENV_NAME="llama-b200"

echo "Creating conda env: $ENV_NAME"
export CONDA_ALWAYS_YES=true
conda create -y -n $ENV_NAME python=3.11
conda activate $ENV_NAME

echo "Python in env: $(which python)"
python --version

################################################################################
# 4. Install PyTorch (CUDA 12.8 / cu128)
################################################################################
echo "Installing PyTorch with CUDA 12.8 (cu128)..."

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

################################################################################
# 5. Install Hugging Face packages
################################################################################
echo "Installing Transformers / Accelerate / HF Hub..."

pip install "transformers>=4.45.0" "accelerate>=0.33.0" "huggingface_hub>=0.24.0"
pip install hf_transfer

################################################################################
# 6. Create GPU Test Script
################################################################################
cat << 'EOF' > gpu_test.py
import os
import torch

print("=== SLURM Info ===")
for key in ["SLURM_JOB_ID", "SLURM_NODELIST", "CUDA_VISIBLE_DEVICES"]:
    print(f"{key}: {os.environ.get(key)}")

print("\n=== PyTorch / CUDA Info ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    print(f"GPU {i}: {name}, compute capability {cap}")

if torch.cuda.is_available():
    x = torch.randn((2048, 2048), device='cuda')
    y = torch.randn((2048, 2048), device='cuda')
    z = x @ y
    print("\nMatmul OK:", z[0,0].item())
else:
    print("\nCUDA not available.")

print("\n=== GPU Test Complete ===")
EOF

################################################################################
# 7. Run GPU Test
################################################################################
echo "Running GPU test..."
python gpu_test.py

echo "===== JOB COMPLETE ====="
