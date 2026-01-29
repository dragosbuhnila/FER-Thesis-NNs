#!/bin/bash
#SBATCH --job-name=dbuhnila/tesi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppp36213@gmail.com
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=100M
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.log
#SBATCH --error=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.log



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

cd /home/dbuhnila/data_backup_260129-155400

echo "Currently in:"
pwd

echo "Calculated size is"
du -sh .