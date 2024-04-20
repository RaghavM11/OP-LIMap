#!/bin/bash

# echo "Installing dependencies..."

# This is really dissatisfying. I believe the issue is that the limap developers didn't specify a
# required version of some of the third-party libraries they use, so the latest versions are
# installed by default, which are incompatible with the limap code. We place these file overrides in
# the limap directory to fix the issue.
echo "Overriding limap files..."
cp -r file_overrides/* .
echo "Done"