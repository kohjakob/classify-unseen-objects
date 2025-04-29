#!/bin/bash

# Check if the PROJECT_ROOT argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <PROJECT_ROOT>"
  exit 1
fi

# Set the PROJECT_ROOT variable
PROJECT_ROOT=$1

# Define the download URL and target paths
DOWNLOAD_URL="https://www.kaggle.com/api/v1/datasets/download/jeremy26/shapenet-core"
ZIP_PATH="$PROJECT_ROOT/classify-unseen-objects/data/shapenetcore/shapenet-core.zip"
UNZIP_DIR="$PROJECT_ROOT/classify-unseen-objects/data/shapenetcore"

# Create the target directory if it doesn't exist
mkdir -p $UNZIP_DIR

# Download the shapenet-core.zip file
curl -L -o $ZIP_PATH $DOWNLOAD_URL

# Unzip the file
unzip $ZIP_PATH -d $UNZIP_DIR

# Delete the zip file
rm $ZIP_PATH

echo "Download and extraction complete."