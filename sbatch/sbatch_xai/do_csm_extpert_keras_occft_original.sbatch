#!/bin/bash
#SBATCH --job-name=dbuh_do_csm_extpert_keras_occft_original
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppp36213@gmail.com
#SBATCH --partition=gpu_a40
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dbuhnila/models/FER-Thesis-NNs/out_err/%j_%x.log
#SBATCH --error=/home/dbuhnila/models/FER-Thesis-NNs/out_err/%j_%x.log



# =====================================================================================
# ============================= Preliminary Setups ====================================
# =====================================================================================

module load miniconda3/3.13.25

source activate base
conda activate fer-thesis

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/dbuhnila/.conda/envs/fer-thesis"

echo "=========================================================================="
echo "Activated venv!"
python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"
echo "=========================================================================="

cd /home/dbuhnila/models/FER-Thesis-NNs
pwd



# =====================================================================================
# =================================== Run Code ========================================
# =====================================================================================


COMMAND_TO_RUN="./scripts/xai/do_csm.py --run_name 20260224-203353_cmplt-run_occft-models_original-testset_do_explainability_extpert_keras"
echo "Now running $COMMAND_TO_RUN"
python $COMMAND_TO_RUN


echo "==================================================================================="
echo "Job ended at:"
python -c "from datetime import datetime; print(datetime.now().isoformat(sep=' ', timespec='seconds'))"


# =====================================================================================
# ============================= Copy Logs Outside =====================================
# =====================================================================================
DEST_ROOT="$HOME/fer-thesis-logs-to-push"
mkdir -p "$DEST_ROOT"

TS=$(date +%Y%m%d-%H%M%S)
DEST_DIR="$DEST_ROOT/job_${SLURM_JOB_ID}_$TS"
mkdir -p "$DEST_DIR"

# Primary log (SLURM --output goes here)
MAIN_LOG="/home/dbuhnila/models/FER-Thesis-NNs/out_err/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log"
# If your job uses timestamped names, adjust accordingly:
if [ -f "$MAIN_LOG" ]; then
  cp "$MAIN_LOG" "$DEST_DIR/" || true
fi

# Also copy the sbatch script used (helpful for provenance)
cp "${SLURM_SUBMIT_DIR:-/home/dbuhnila/models/FER-Thesis-NNs}/sbatch/evaluate_model_gpu.sbatch" "$DEST_DIR/" 2>/dev/null || true

# Create a zip with all the code inside of scripts and modules
# Prefer zip if available, otherwise fall back to tar.gz
CODE_ARCHIVE="$DEST_DIR/code_$(date +%Y%m%d-%H%M%S).zip"
ROOT_DIR="/home/dbuhnila/models/FER-Thesis-NNs"
if command -v zip >/dev/null 2>&1; then
  echo "Creating ZIP archive $CODE_ARCHIVE (scripts/ + modules/)"
  (cd "$ROOT_DIR" && zip -r "$CODE_ARCHIVE" scripts modules -x "*.git*" "*/__pycache__/*") >/dev/null 2>&1 || echo "zip failed"
else
  # fallback to tar.gz
  CODE_TAR="${CODE_ARCHIVE%.zip}.tar.gz"
  echo "zip not found, creating tarball $CODE_TAR"
  tar -C "$ROOT_DIR" -czf "$CODE_TAR" scripts modules --exclude='.git' --exclude='*/__pycache__' || echo "tar failed"
fi

# Create a provenance file
PROV="$DEST_DIR/provenance.txt"
{
  echo "JOBID: ${SLURM_JOB_ID}"
  echo "USER: ${USER}"
  echo "HOST: $(hostname)"
  echo "DATE: $(date --utc)"
  echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-unknown}"
  echo "PYTHON_EXECUTABLE: $(which python || echo unknown)"
  echo "GIT_COMMIT: $(git -C /home/dbuhnila/models/FER-Thesis-NNs rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"
} > "$PROV"

# Done - show where files are
echo "Saved logs and snapshot to $DEST_DIR"