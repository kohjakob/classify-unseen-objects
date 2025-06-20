#!/bin/bash

# Check if the PROJECT_ROOT, start_scene, and stop_scene arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <PROJECT_ROOT> <start_scene> <stop_scene>"
  echo "Example: $0 /path/to/project/root 0 49"
  exit 1
fi

# Set the PROJECT_ROOT and output directory variables
PROJECT_ROOT=$1
OUTPUT_DIR="$PROJECT_ROOT/classify-unseen-objects/data/scannet/scannet_scenes"
SENSREADER_DIR="$PROJECT_ROOT/classify-unseen-objects/external/ScanNet/SensReader/python"

# Get the start and stop scene arguments
START_SCENE=$2
STOP_SCENE=$3

# Loop through the scene IDs and download the `.sens` files
for i in $(seq -w $START_SCENE $STOP_SCENE)
do
    SCENE_ID="scene$(printf "%04d" $i)_00"
    echo "Processing .sens file for ${SCENE_ID}..."

    # Construct the expected file path
    SENS_FILE="${OUTPUT_DIR}/${SCENE_ID}/${SCENE_ID}.sens"

    # Check if the .sens file already exists
    if [ -f "$SENS_FILE" ]; then
        echo "  File ${SENS_FILE} already exists. Skipping download."
    else
        echo "  File ${SENS_FILE} is missing."
    fi

    # Extract posed images from the .sens file
    POSED_IMAGES_DIR="${OUTPUT_DIR}/${SCENE_ID}/posed_images"
    if [ -f "$SENS_FILE" ]; then
        if [ ! -d "$POSED_IMAGES_DIR/intrinsic" ]; then
            echo "  Extracting posed images from ${SENS_FILE} into ${POSED_IMAGES_DIR}..."
            mkdir -p "$POSED_IMAGES_DIR"
            python2 $SENSREADER_DIR/reader.py \
                --filename "$SENS_FILE" \
                --output_path "$POSED_IMAGES_DIR" \
                --export_color_images \
                --export_depth_images \
                --export_poses \
                --export_intrinsics
        else
            echo "  Posed images already extracted for ${SCENE_ID}. Skipping extraction."
        fi
    else
        echo "  .sens file not found for ${SCENE_ID}. Skipping posed image extraction."
    fi
done