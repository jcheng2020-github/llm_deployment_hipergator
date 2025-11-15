#!/bin/bash
#SBATCH --job-name=b200_gpu_test
#SBATCH --time=0:30:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=b200_gpu_test_%j.out

echo "===== JOB START ====="
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

###############################################################################
# 1. Load CUDA + conda
###############################################################################
module load cuda/12.8.1
module load conda          # adjust if your site uses a different module name

# Initialize conda shell
source "$(conda info --base)/etc/profile.d/conda.sh"

###############################################################################
# 2. Activate existing environment
###############################################################################
ENV_NAME="llama-b200"
echo "Activating conda env: $ENV_NAME"
conda activate "$ENV_NAME"

echo "Python: $(which python)"
python --version
echo "Loaded modules:"
module list 2>&1

###############################################################################
# 3. Python GPU test (checks CUDA + GPU type)
###############################################################################
python - << 'EOF'
import os
import sys
import torch

print("\n=== SLURM Info ===")
for k in ["SLURM_JOB_ID", "SLURM_NODELIST", "CUDA_VISIBLE_DEVICES"]:
    print(f"{k}: {os.environ.get(k)}")

print("\n=== PyTorch / CUDA Info ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("ERROR: CUDA not available in this job.")
    sys.exit(1)

num_gpus = torch.cuda.device_count()
print("GPU count:", num_gpus)

if num_gpus == 0:
    print("ERROR: No GPUs visible to PyTorch.")
    sys.exit(1)

for i in range(num_gpus):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    print(f"GPU {i}: {name}, compute capability {cap}")

# Basic GPU type check – look for 'b200' in the name
name0 = torch.cuda.get_device_name(0)
if "b200" in name0.lower():
    print("\nGPU type check: PASS (looks like a B200 GPU).")
else:
    print(f"\nGPU type check: WARNING – name '{name0}' does not contain 'B200'.")
    print("Verify that the job is really on a B200 node/partition.")

# Simple matmul test
print("\nRunning small matmul on GPU 0...")
x = torch.randn((2048, 2048), device="cuda:0")
y = torch.randn((2048, 2048), device="cuda:0")
z = x @ y
print("Matmul OK, sample value:", float(z[0, 0]))

print("\n=== GPU Test Complete ===")
EOF

echo "===== JOB END ====="
