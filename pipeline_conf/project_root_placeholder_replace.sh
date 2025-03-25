#!/bin/bash

# Check if the PROJECT_ROOT argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <PROJECT_ROOT>"
  exit 1
fi

# Set the PROJECT_ROOT variable
PROJECT_ROOT=$1

# Define the placeholder and the actual path
PLACEHOLDER="PLACEHOLDER_PROJECT_ROOT"
ACTUAL_PATH="$PROJECT_ROOT"

# Replace the placeholder with the actual path in all .yml files
find $PROJECT_ROOT -type f -name "*.yml" -exec sed -i "s|$PLACEHOLDER|$ACTUAL_PATH|g" {} +
find $PROJECT_ROOT -type f -name "*.py" -exec sed -i "s|$PLACEHOLDER|$ACTUAL_PATH|g" {} +

echo "Placeholders replaced with actual paths."