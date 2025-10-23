<!-- Current Instructions -->
conda create -n fer-thesis python=3.11 -y
conda activate fer-thesis

python -m pip install --upgrade pip setuptools wheel
python -m pip install tensorflow-io-gcs-filesystem==0.31.0
python -m pip install tensorflow==2.15.0
python -m pip install h5py "numpy>=1.23.5,<2.0.0" Pillow scikit-learn ultralytics


conda activate fer-thesis
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"

<!-- Old Insructions (what we had when using poetry) -->

poetry init -n --python ">=3.11,<3.12"
poetry config virtualenvs.in-project true --local
poetry env use "C:\Users\Dragos\AppData\Local\Programs\Python\Python311\python.exe"
poetry install --no-root

poetry add tensorflow-io-gcs-filesystem==0.31.0
poetry add tensorflow==2.15.0
poetry run pip install tensorflow==2.15.0
poetry add h5py numpy==">=1.23.5,<2.0.0" Pillow scikit-learn

<!-- pyproject.toml:
python = ">=3.11,<3.12"
tensorflow-io-gcs-filesystem = "0.31.0"
tensorflow = "2.15.0"
h5py = "^3.15.0"
numpy = ">=1.23.5,<2.0.0"
pillow = "^11.3.0"
scikit-learn = "^1.7.2"
ultralytics = "^8.3.218" -->

<!--  -->