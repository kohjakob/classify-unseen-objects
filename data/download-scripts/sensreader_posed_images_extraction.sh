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

# Loop through the scene IDs and process the `.sens` files
for i in $(seq -w $START_SCENE $STOP_SCENE)
do
    SCENE_ID="scene$(printf "%04d" $i)_00"
    echo "Processing .sens file for ${SCENE_ID}..."

    # Construct the expected file path
    SENS_FILE="${OUTPUT_DIR}/${SCENE_ID}/${SCENE_ID}.sens"

    # Check if the .sens file already exists
    if [ ! -f "$SENS_FILE" ]; then
        echo "  File ${SENS_FILE} is missing. Skipping extraction."
        continue
    fi

    # Define directory paths
    POSED_IMAGES_DIR="${OUTPUT_DIR}/${SCENE_ID}/posed_images"
    COLOR_DIR="${POSED_IMAGES_DIR}/color"
    DEPTH_DIR="${POSED_IMAGES_DIR}/depth" 
    POSE_DIR="${POSED_IMAGES_DIR}/pose"
    INTRINSIC_DIR="${POSED_IMAGES_DIR}/intrinsic"
    
    # Make sure the main directory exists
    mkdir -p "$POSED_IMAGES_DIR"
    
    # Extract color images if needed
    if [ ! -d "$COLOR_DIR" ] || [ -z "$(ls -A $COLOR_DIR 2>/dev/null)" ]; then
        echo "  Extracting color images..."
        mkdir -p "$COLOR_DIR"
        python2 $SENSREADER_DIR/reader.py \
            --filename "$SENS_FILE" \
            --output_path "$POSED_IMAGES_DIR" \
            --export_color_images
    else
        echo "  Color images already exist. Skipping."
    fi
    
    # Extract depth images if needed
    if [ ! -d "$DEPTH_DIR" ] || [ -z "$(ls -A $DEPTH_DIR 2>/dev/null)" ]; then
        echo "  Extracting depth images..."
        mkdir -p "$DEPTH_DIR"
        python2 $SENSREADER_DIR/reader.py \
            --filename "$SENS_FILE" \
            --output_path "$POSED_IMAGES_DIR" \
            --export_depth_images
    else
        echo "  Depth images already exist. Skipping."
    fi
    
    # Extract poses if needed
    if [ ! -d "$POSE_DIR" ] || [ -z "$(ls -A $POSE_DIR 2>/dev/null)" ]; then
        echo "  Extracting poses..."
        mkdir -p "$POSE_DIR"
        python2 $SENSREADER_DIR/reader.py \
            --filename "$SENS_FILE" \
            --output_path "$POSED_IMAGES_DIR" \
            --export_poses
    else
        echo "  Poses already exist. Skipping."
    fi
    
    # Extract intrinsics if needed
    if [ ! -d "$INTRINSIC_DIR" ] || [ ! -f "${INTRINSIC_DIR}/intrinsic_color.txt" ] || [ ! -f "${INTRINSIC_DIR}/intrinsic_depth.txt" ]; then
        echo "  Extracting intrinsics..."
        mkdir -p "$INTRINSIC_DIR"
        python2 $SENSREADER_DIR/reader.py \
            --filename "$SENS_FILE" \
            --output_path "$POSED_IMAGES_DIR" \
            --export_intrinsics
    else
        echo "  Intrinsics already exist. Skipping."
    fi
    
    echo "  Processing complete for ${SCENE_ID}."
done