#!/usr/bin/env bash
set -e

BUILD_TYPE=Release
NUM_PROC=4

BASE_DIR="$PWD"

echo "Base directory: $BASE_DIR"

# Build main project
cd "$BASE_DIR"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j"$NUM_PROC"
cd ..
