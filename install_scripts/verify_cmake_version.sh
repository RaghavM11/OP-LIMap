#!/bin/bash

# Function for checking the CMake version
verify_cmake_version() {
    echo "Checking CMake version..."
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

verify_cmake_version