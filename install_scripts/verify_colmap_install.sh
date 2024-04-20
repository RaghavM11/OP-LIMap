#!/bin/bash

# Don't want to set this because this could fail and still be okay since this script leaves
# something to be desired.
# set -e

# Define a function to check if a CMake package can be found
check_package_with_cmake() {
  local package_name=$1

  echo "Attempting to find CMake package '${package_name}' with hacky script"

  # Create a temporary CMake script
  local cmake_script=$(mktemp /tmp/find_package_test.XXXXXX.cmake)

  # Write the find_package command to the script

  # echo "set(CMAKE_CXX_STANDARD 11)" >> $cmake_script
  # echo "project(ColmapFinder C CXX)" >> $cmake_script
  echo "find_package(${package_name} REQUIRED)" >> $cmake_script

  # Run cmake script in script mode (-P)
  cmake -P $cmake_script > /dev/null 2>&1
  local return_code=$?
  # echo "Cmake script output: $(cmake -P $cmake_script)"


  # Clean up temporary CMake script
  rm $cmake_script

  # Check return code from cmake command
  if [ $return_code -eq 0 ]; then
    echo "Package '${package_name}' found by CMake."
    return 0
  else
    echo
    echo "Package '${package_name}' not found by hacky CMake scripting. This doesn't necessarily"
    echo "mean that the package isn't installed, but it's something to look at if installation"
    echo "fails."
    echo
    return 0
  fi
}

# Example usage of the function
# package_name="SomePackage" # Replace this with the actual package name
# check_package_with_cmake "COLMAP"
check_package_with_cmake $1

