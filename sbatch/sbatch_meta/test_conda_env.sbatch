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

module load miniconda3/3.13.25

# # Activate Conda environment
source activate base
conda activate fer-thesis

# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# echo "=========================================================================="
# echo "Activated venv!"
# python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
# python -c "import tensorflow as tf; print('GPUs available:' + str(tf.config.list_physical_devices('GPU')) if tf.config.list_physical_devices('GPU') else 'No GPUs available')"
# echo "=========================================================================="

# nvcc --version
# cat $CONDA_PREFIX/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

conda list