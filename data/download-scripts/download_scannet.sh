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
)

# Loop through the scene IDs and download the specified file types
for i in $(seq -w $START_SCENE $STOP_SCENE)
do
    SCENE_ID="scene$(printf "%04d" $i)_00"
    echo "Downloading files for ${SCENE_ID}..."

    for TYPE in "${FILE_TYPES[@]}"
    do
        echo "  Downloading ${TYPE}..."
        echo | python3 $PROJECT_ROOT/classify-unseen-objects/data/download-scripts/download_scannet.py --type $TYPE --id $SCENE_ID --o $OUTPUT_DIR
    done
done