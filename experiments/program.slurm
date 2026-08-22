#!/bin/bash
#SBATCH -A naiss2026-4-1340-gpu
#SBATCH -t 10:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH --gres=gpu:1
#SBATCH -J dlnam_gpu
#SBATCH --mail-user=shivang.pandey@ki.se
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

set -euo pipefail

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

CONTAINER="/nobackup/proj/disk/dlnam/shared/containers/dlnam_container_v1.sif"

SCRIPT="experiments/test.py"

echo "============================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $(hostname)"
echo "Allocated CPUs:  ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Container:       $CONTAINER"
echo "Python script:   $SCRIPT"
echo "Start:           $(date)"
echo "============================================================"

# ------------------------------------------------------------
# Check required files
# ------------------------------------------------------------

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: Container not found:"
    echo "$CONTAINER"
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Python script not found:"
    echo "$SCRIPT"
    exit 1
fi

# ------------------------------------------------------------
# GPU information
# ------------------------------------------------------------

echo
echo "GPU information:"
nvidia-smi

# ------------------------------------------------------------
# Validate the container
# ------------------------------------------------------------

echo
echo "Validating container:"
apptainer sif list "$CONTAINER" >/dev/null
ls -lh "$CONTAINER"

# ------------------------------------------------------------
# Test PyTorch and CUDA
# ------------------------------------------------------------

echo
echo "Testing PyTorch and CUDA..."

apptainer exec \
    --cleanenv \
    --nv \
    "$CONTAINER" \
    python -u - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the container")

print("GPU:", torch.cuda.get_device_name(0))
PY

# ------------------------------------------------------------
# Run DLNAM analysis
# ------------------------------------------------------------

echo
echo "Starting DLNAM analysis..."

apptainer exec \
    --cleanenv \
    --nv \
    --bind /nobackup:/nobackup \
    "$CONTAINER" \
    python -u "$SCRIPT"

echo
echo "Completed: $(date)"
