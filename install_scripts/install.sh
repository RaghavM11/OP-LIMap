#!/bin/bash

set -e

# Verify that the directory with which this bash script was executed from is the LIMap-Extension directory
if [[ "$(basename "$(pwd)")" != "LIMap-Extension" ]]; then
    echo "Please execute this script from the LIMap-Extension directory."
    exit 1
fi

./install_scripts/verify_cmake_version.sh

echo "Ensuring submodules are up to date..."
git submodule update --init --recursive

# Check that colmap can be found with CMake's find_package macro
./install_scripts/verify_colmap_install.sh COLMAP
./install_scripts/verify_colmap_install.sh PoseLib

python3 install_scripts/verify_apt_installations.py \
    libhdf5-dev \
    libopencv-dev \
    libopencv-contrib-dev \
    libarpack++2-dev \
    libarpack2-dev \
    libsuperlu-dev \
    git \
    curl

if [[ -z "$(which asdf)" ]]; then
    echo "Installing asdf..."
    git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.14.0
    echo '. $HOME/.asdf/asdf.sh' >> ~/.bashrc
    echo '. $HOME/.asdf/completions/asdf.bash' >> ~/.bashrc
    source ~/.bashrc
    asdf plugin add python
else
    echo "asdf already installed."
fi

echo "Installing pipenv..."
if [[ -z "$(which pipenv)" ]]; then
    pip3 install pipenv --user
else
    echo "pipenv already installed."
fi

echo "Installing dependencies (ASSUMING CUDA 11.6 CAPABLE GPU)..."
echo "    Installation should work without GPU support, but it is untested."
echo "    You will need to edit the Pipfile to switch to CPU -based torch and torchvision index URLs"
# I could be wrong, but I think the --site-packages flag is necessary to allow the limap build to
# use the system's OpenCV installation. But I'm also skeptical as we only installed OpenCV
# libraries, not the Python packages, via apt.
pipenv install 3.9 --site-packages

# This is really dissatisfying. I believe the issue is that the limap developers didn't specify a
# required version of some of the third-party libraries they use, so the latest versions are
# installed by default, which are incompatible with the limap code. We place these file overrides in
# the limap directory to fix the issue.
echo "Overriding limap files..."
cp -r file_overrides/* .
echo "Done"
