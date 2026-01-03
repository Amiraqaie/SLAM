#!/bin/bash

# Path to the binary
PROGRAM="/home/amir/ImageProcessing/KITTY/FinalProject/bin/run_kitti_stereo"

# Arguments to pass to the program (empty for now)
ARGS=()  # You can add your arguments here

# Set the working directory to the VS Code workspace folder
cd "/home/amir/ImageProcessing/KITTY/FinalProject/" || exit 1

# Run with GDB
gdb --args "$PROGRAM" "${ARGS[@]}"
