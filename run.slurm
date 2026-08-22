#!/bin/bash

#SBATCH -A naiss2026-4-1340-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH --gpus=1
#SBATCH -t 20:00:00
#SBATCH -J dlnam_gpu

#SBATCH --mail-user=shivang.pandey@ki.se
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err

set -euo pipefail

CONTAINER="/nobackup/proj/disk/k9_climes/personal/shipan/containers/dlnam_container_arm64_v2.sif"
SCRIPT="experiments/run_mc_joint.py"

echo "============================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $(hostname)"
echo "Partition:       ${SLURM_JOB_PARTITION:-unknown}"
echo "Allocated CPUs:  ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Allocated GPUs:  ${SLURM_GPUS:-unknown}"
echo "Container:       $CONTAINER"
echo "Python script:   $SCRIPT"
echo "Architecture:    $(uname -m)"
echo "Start:           $(date)"
echo "============================================================"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Python script not found: $SCRIPT"
    exit 1
fi

echo
echo "GPU information:"
nvidia-smi

echo
echo "Container information:"
ls -lh "$CONTAINER"
apptainer sif list "$CONTAINER" >/dev/null

echo
echo "Testing PyTorch and CUDA..."

srun --ntasks=1 apptainer exec \
    --cleanenv \
    --nv \
    --bind /nobackup:/nobackup \
    "$CONTAINER" \
    python -u - <<'PY'
import platform
import torch

print("Machine architecture:", platform.machine())
print("PyTorch version:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the container")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "Starting DLNAM analysis..."

srun --ntasks=1 apptainer exec \
    --cleanenv \
    --nv \
    --bind /nobackup:/nobackup \
    "$CONTAINER" \
    python -u "$SCRIPT"

echo
echo "Completed: $(date)"
