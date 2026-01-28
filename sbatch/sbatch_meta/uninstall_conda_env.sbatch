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

conda env remove -n fer-thesis

rm -rf /home/dbuhnila/.conda/envs/fer-thesis

source activate base
conda activate fer-thesis
echo "=========================================================================="
echo "Activated venv!"
python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"
python -c "import torch; print('PyTorch GPUs available:' + str(torch.cuda.device_count()) if torch.cuda.is_available() else 'No GPUs available')"
echo "=========================================================================="

cd /home/dbuhnila/models/FER-Thesis-NNs
pwd


