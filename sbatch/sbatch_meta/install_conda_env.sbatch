#!/bin/bash
#SBATCH --job-name=dbuh_install_conda_env
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppp36213@gmail.com
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --mem=100M
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/dbuhnila/models/FER-Thesis-NNs/out_err/%j_%x.log
#SBATCH --error=/home/dbuhnila/models/FER-Thesis-NNs/out_err/%j_%x.log

module load miniconda3/3.13.25

# confirm conda is found
which conda
conda --version

# accept Anaconda channels' ToS (replace CHANNEL with the shown URLs)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


conda create -n fer-thesis python=3.11 -y
echo "=========================================================================="
echo "Created venv!"
echo "=========================================================================="

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

conda install -y nvidia/label/cuda-12.2.0::cuda-toolkit
echo "=========================================================================="
echo "Installed cudatoolkit 12.2"
echo "=========================================================================="

conda install -y cudnn=8.9 -c conda-forge
echo "=========================================================================="
echo "Installed cudnn 8.9"
echo "=========================================================================="

python -m pip install tensorflow-io-gcs-filesystem==0.31.0
echo "=========================================================================="
echo "Installed tf io-gs-fs!"
echo "=========================================================================="

python -m pip install tensorflow==2.15.0
echo "=========================================================================="
echo "Installed tf!"
echo "=========================================================================="


# protobuf should be 4.25.8 tho
python -m pip install h5py "numpy>=1.23.5,<2.0.0" Pillow scikit-learn ultralytics gdown mediapipe mlflow waitress argparse rich seaborn reportlab
python -m pip install --force-reinstall --no-deps "protobuf==4.25.8"
echo "=========================================================================="
echo "Installed other dependencies! (h5py, numpy, Pillow, scikit-learn, ultralytics, gdown, mediapipe, mlflow, waitress, argparse, rich, seaborn, reportlab)"
echo "=========================================================================="


python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"