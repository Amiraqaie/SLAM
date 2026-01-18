#!/usr/bin/env bash
set -e

BUILD_TYPE=Release
NUM_PROC=4

BASE_DIR="$PWD"

echo "Base directory: $BASE_DIR"

# Build main project
cd "$BASE_DIR"
mkdir -p tmp
cd tmp
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_CXX_FLAGS="-std=c++11 -Wall" -DCMAKE_CXX_FLAGS_RELEASE="-std=c++17 -O3 -fopenmp -pthread" ..
make -j"$NUM_PROC" 
./run_kitti_stereo
