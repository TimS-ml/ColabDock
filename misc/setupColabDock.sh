#!/bin/bash

# [1] clone colabdock && install packages
# git clone https://github.com/TimS-ml/ColabDock.git
apt-get update
apt-get install -y aria2 python3.8-venv

# [2] setup virtual env
python3.8 -m venv venv
source venv/bin/activate
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.8+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl
pip install jax==0.3.8
pip install -r ./ColabDock/requirements.txt
# pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install --upgrade dm-haiku  # fix jax.config.jax_experimental_name_stack

# [3] setup AlphaFold weights
cd ColabDock
mkdir params
aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf alphafold_params_2022-12-06.tar -C params

# [4] update config.py
sed -i 's|/path/to/alphafold|./params/|g' config.py

# [5] run
python main.py
