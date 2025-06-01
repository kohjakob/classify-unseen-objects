#!/bin/bash

# Check if the PROJECT_ROOT, start_scene, and stop_scene arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <PROJECT_ROOT> <start_scene> <stop_scene>"
  echo "Example: $0 /path/to/project/root 0 49"
  exit 1
fi

# Set the PROJECT_ROOT and output directory variables
PROJECT_ROOT=$1
OUTPUT_DIR="$PROJECT_ROOT/classify-unseen-objects/data/scannet/scannet_scenes/"

# Get the start and stop scene arguments
START_SCENE=$2
STOP_SCENE=$3

# Define the file types to download
FILE_TYPES=(
    "_vh_clean.aggregation.json"
    "_vh_clean.ply"
    "_vh_clean.segs.json"
    "_vh_clean_2.ply"                     # Main 3D point cloud file
    "_vh_clean_2.0.010000.segs.json"      # Superpoint segmentation file
    "_2d-instance.zip"                    # 2D instance masks
    ".txt"                                # Camera pose and intrinsic files
    ".sens"                               # Raw RGB-D data
)

# Loop through the scene IDs and download the specified file types
for i in $(seq -w $START_SCENE $STOP_SCENE)
do
    SCENE_ID="scene$(printf "%04d" $i)_00"
    echo "Processing files for ${SCENE_ID}..."

    for TYPE in "${FILE_TYPES[@]}"
    do
        # Construct the expected file path
        FILE_PATH="${OUTPUT_DIR}/${SCENE_ID}/${SCENE_ID}${TYPE}"

        # Check if the file already exists
        if [ -f "$FILE_PATH" ]; then
            echo "  File ${FILE_PATH} already exists. Skipping download."
        else
            echo "  Downloading ${TYPE} for ${SCENE_ID}..."
            
            # Automatically handle the key press for `.sens` files
            if [[ "$TYPE" == ".sens" ]]; then
                echo -e "\n" | python3 $PROJECT_ROOT/classify-unseen-objects/data/download-scripts/download_scannet.py --type $TYPE --id $SCENE_ID --o $OUTPUT_DIR
            else
                echo | python3 $PROJECT_ROOT/classify-unseen-objects/data/download-scripts/download_scannet.py --type $TYPE --id $SCENE_ID --o $OUTPUT_DIR
            fi
        fi

        # If the file is a zip file, unzip it
        if [[ "$TYPE" == "_2d-instance.zip" ]]; then
            UNZIP_DIR="${OUTPUT_DIR}/${SCENE_ID}/posed_images"
            if [ ! -d "$UNZIP_DIR/instance" ]; then
                echo "  Unzipping ${FILE_PATH} to ${UNZIP_DIR}..."
                mkdir -p "$UNZIP_DIR"
                unzip -q "$FILE_PATH" -d "$UNZIP_DIR"
            else
                echo "  ${UNZIP_DIR} already exists. Skipping unzip."
            fi
        fi
    done
done