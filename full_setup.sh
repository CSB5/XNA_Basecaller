#!/bin/bash
set -e # Exit on first error encountered
set -x # Print commands

# Download Minimap or create a symbolic link to it at bin/minimap2:
curl -L https://github.com/lh3/minimap2/releases/download/v2.17/minimap2-2.17_x64-linux.tar.bz2 | tar -jxvf - minimap2-2.17_x64-linux/minimap2
mv -v minimap2-2.17_x64-linux bin

# Install python enviroment
conda env create -f env.yml
conda activate xna_bc

# Install UB-Bonito, default install with cuda 10.2
cd ub-bonito/
python3 -m venv venv3
source venv3/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# pip install -r requirements-cuda111.txt --extra-index-url https://download.pytorch.org/whl/cu111/
# pip install -r requirements-cuda113.txt --extra-index-url https://download.pytorch.org/whl/cu113/
python setup.py develop
deactivate
cd ..

# Download Data
bash ./download_data.sh