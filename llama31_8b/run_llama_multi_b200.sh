#!/bin/bash
#SBATCH --job-name=llama31_8b_multi
#SBATCH --time=0:30:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2          # <--- number of B200 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=llama31_8b_multi_%j.out

echo "===== JOB START ====="
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Working dir: $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR"

###############################################################################
# 1. Load CUDA + conda
###############################################################################
module load cuda/12.8.1
module load conda

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama-b200

echo "Python: $(which python)"
python -V
echo "Modules:"
module list 2>&1

###############################################################################
# 2. Hugging Face cache inside working dir
###############################################################################
WORKDIR="$SLURM_SUBMIT_DIR"
export HF_TOKEN=$(cat ~/.secrets/hf_token)
export HF_HOME="$WORKDIR/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_EVALUATE_CACHE="$HF_HOME/evaluate"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_EVALUATE_CACHE"

echo "HF_HOME: $HF_HOME"

###############################################################################
# 3. Start GPU monitor (nvidia-smi every 60 seconds)
###############################################################################
MONITOR_LOG="$WORKDIR/gpu_monitor_${SLURM_JOB_ID}.log"
echo "Starting GPU monitor; logging to $MONITOR_LOG"
(
  while true; do
    echo "----- $(date) -----" >> "$MONITOR_LOG"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
               --format=csv,noheader,nounits >> "$MONITOR_LOG"
    echo "" >> "$MONITOR_LOG"
    sleep 60
  done
) &
MONITOR_PID=$!
echo "Monitor PID: $MONITOR_PID"

###############################################################################
# 4. Run Llama script
###############################################################################
set -e
srun python run_llama_multi_chat.py
STATUS=$?

###############################################################################
# 5. Stop monitor and finish
###############################################################################
echo "Stopping GPU monitor..."
kill $MONITOR_PID 2>/dev/null || echo "Monitor already stopped."

echo "Job exit status: $STATUS"
echo "===== JOB END ====="
exit $STATUS
