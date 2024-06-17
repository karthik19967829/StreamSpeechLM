wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda create -n cyborg-env python=3.11
conda activate cyborg-env
pip install -r requirements.txt
wget https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip
apt-get update
apt-get install unzip
unzip exp.zip
