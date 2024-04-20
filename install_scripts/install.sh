#!/bin/bash

# Verify that the directory with which this bash script was executed from is the LIMap-Extension directory
if [[ "$(basename "$(pwd)")" != "LIMap-Extension" ]]; then
    echo "Please execute this script from the LIMap-Extension directory."
    exit 1
fi

./install_scripts/verify_cmake_version.sh


echo "Ensuring submodules are up to date..."
git submodule update --init --recursive

# TODO: Ensure colmap and poselib are installed.

# TODO: Parse apt list --installed to ensure apt dependencies are installed
# Perhaps by using a Python file that raises an exception if they aren't?
# apt dependencies:
# - libhdf5-dev
# - libopencv-dev
# - libopencv-contrib-dev
# - libarpack++2-dev
# - libarpack2-dev
# - libsuperlu-dev

# Check that colmap can be found with CMake's find_package macro
./install_scripts/verify_colmap_install.sh

# echo "Installing dependencies..."

# This is really dissatisfying. I believe the issue is that the limap developers didn't specify a
# required version of some of the third-party libraries they use, so the latest versions are
# installed by default, which are incompatible with the limap code. We place these file overrides in
# the limap directory to fix the issue.
echo "Overriding limap files..."
cp -r file_overrides/* .
echo "Done"
