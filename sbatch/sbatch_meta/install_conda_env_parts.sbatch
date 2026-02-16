#!/bin/bash
#SBATCH --job-name=dbuhnila/tesi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppp36213@gmail.com
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --mem=100M
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.log
#SBATCH --error=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.log

module load miniconda3/3.13.25

source activate base
conda activate fer-thesis
echo "=========================================================================="
echo "Activated venv!"
python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
echo "=========================================================================="

python -m pip install --upgrade pip setuptools wheel
echo "=========================================================================="
echo "Upgraded pip!"
echo "=========================================================================="

python -m pip install rich
echo "=========================================================================="
echo "Installed other dependencies! (rich)"
echo "=========================================================================="

python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"