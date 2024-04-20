#!/bin/bash

# Function for checking the CMake version
verify_cmake_version() {
    CMAKE_VERSION=$(cmake --version)

    # Use grep with Perl-compatible regular expressions (-P) and only output the matched version number (-o)
    kREGEX_EXPRESSION='version\s+\K[0-9]+\.[0-9]+(\.[0-9]+)?'
    version=$(echo "$CMAKE_VERSION" | grep -Po 'version\s+\K[0-9]+\.[0-9]+(\.[0-9]+)?')
    if [[ $version ]]; then
        # Save the current IFS value so it can be restored later
        OLDIFS=$IFS
        # Set IFS to the delimiter (.)
        IFS='.'
        # Read the version into three separate variables
        read -r CMAKE_MAJOR CMAKE_MINOR CMAKE_BUGFIX <<< "$version"
        # Restore the original IFS value
        IFS=$OLDIFS

        if [[ $CMAKE_MAJOR -lt 3 ]]; then
            echo "CMake version is less than 3.17. Please install CMake version 3.17 or greater."
            exit 1
        elif [[ $CMAKE_MAJOR -eq 3 && $CMAKE_MINOR -lt 17 ]]; then
            echo "CMake version is less than 3.17. Please install CMake version 3.17 or greater."
            exit 1
        else
            echo "CMake version is 3.17 or greater, CMake dependency is met."
        fi
    else
        echo "CMake version could not be determined."
        exit 1
    fi
}

# Verify that the directory with which this bash script was executed from is the LIMap-Extension directory
if [[ "$(basename "$(pwd)")" != "LIMap-Extension" ]]; then
    echo "Please execute this script from the LIMap-Extension directory."
    exit 1
fi

echo "Ensuring submodules are up to date..."
git submodule update --init --recursive

# TODO: Ensure colmap and poselib are installed.

# TODO: Parse apt list --installed to ensure apt dependencies are installed
# Perhaps by using a Python file that raises an exception if they aren't?

# TODO: Ensure CMAKE version is at least 3.17
verify_cmake_version

# echo "Installing dependencies..."

# This is really dissatisfying. I believe the issue is that the limap developers didn't specify a
# required version of some of the third-party libraries they use, so the latest versions are
# installed by default, which are incompatible with the limap code. We place these file overrides in
# the limap directory to fix the issue.
echo "Overriding limap files..."
cp -r file_overrides/* .
echo "Done"