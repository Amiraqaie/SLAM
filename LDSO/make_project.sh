#!/usr/bin/env bash
set -e

BUILD_TYPE=Release
NUM_PROC=4

BASE_DIR="$PWD"

echo "Base directory: $BASE_DIR"

# Build DBOW3
cd "$BASE_DIR/thirdparty/DBow3"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j"$NUM_PROC"
sudo make install

# Build g2o
cd "$BASE_DIR/thirdparty/g2o"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j"$NUM_PROC"
sudo make install

# Build Pangolin
cd "$BASE_DIR/thirdparty/Pangolin"
./scripts/install_prerequisites.sh recommended
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j"$NUM_PROC"
sudo make install

# Build main project
cd "$BASE_DIR"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j"$NUM_PROC"
