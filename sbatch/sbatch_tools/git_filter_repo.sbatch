#!/bin/bash
#SBATCH --job-name=dbuhnila/tesi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppp36213@gmail.com
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=100M
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/dbuhnila/models/FER-Thesis-NNs/out_err/%j_%x.log
#SBATCH --error=/home/dbuhnila/models/FER-Thesis-NNs/out_err/%j_%x.log



# =====================================================================================
# ============================= Preliminary Setups ====================================
# =====================================================================================

module load miniconda3/3.13.25

source activate base
conda activate fer-thesis

echo "=========================================================================="
echo "Activated venv!"
echo "=========================================================================="

cd /home/dbuhnila/models/FER-Thesis-NNs
pwd



# =====================================================================================
# =================================== Run Code ========================================
# =====================================================================================

pip install git-filter-repo

git filter-repo --path results_light/debugging_hashes/20260119-122848__landmark_coords_dict.npy --invert-paths