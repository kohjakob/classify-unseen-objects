#!/bin/bash

# Check if the PROJECT_ROOT argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <PROJECT_ROOT>"
  exit 1
fi

# Set the PROJECT_ROOT variable
PROJECT_ROOT=$1

# Define the target subdirectory
TARGET_SUBDIR="$PROJECT_ROOT/classify-unseen-objects"

# Ensure the target subdirectory exists
if [ ! -d "$TARGET_SUBDIR" ]; then
  echo "Error: Directory $TARGET_SUBDIR does not exist."
  exit 1
fi

# Define the marker comment
MARKER_COMMENT="# PROJECT_ROOT_VARIABLE_MARKER"

# Find all Python and YAML files, but only within the classify-unseen-objects subdirectory
find "$TARGET_SUBDIR" -type f \( -name "*.py" -o -name "*.yml" -o -name "*.yaml" \) | while read -r file; do
  
  # First check if the file contains the marker comment to avoid processing files unnecessarily
  if grep -q "$MARKER_COMMENT" "$file"; then
    echo "Found marker in file: $file"
    
    # Use a temporary file
    temp_file=$(mktemp)
    
    # Flag to indicate if we just saw a marker
    found_marker=0
    
    # Process the file line by line
    while IFS= read -r line || [[ -n "$line" ]]; do
      # If we found a marker in the previous iteration
      if [ $found_marker -eq 1 ]; then
        # Extract indentation and variable name
        if [[ "$line" =~ ^([[:space:]]*)([a-zA-Z0-9_]+)(.*) ]]; then
          indentation="${BASH_REMATCH[1]}"
          var_name="${BASH_REMATCH[2]}"
          
          # Write the new line with updated path
          echo "${indentation}${var_name} = \"$PROJECT_ROOT\"" >> "$temp_file"
        else
          # If we can't parse the line, keep it as is
          echo "$line" >> "$temp_file"
        fi
        
        # Reset the marker flag
        found_marker=0
      else
        # Write the current line
        echo "$line" >> "$temp_file"
        
        # Check if this line contains the marker
        if [[ "$line" == *"$MARKER_COMMENT"* ]]; then
          found_marker=1
        fi
      fi
    done < "$file"
    
    # Compare the temporary file with the original before replacing
    if ! diff -q "$temp_file" "$file" > /dev/null; then
      # Only replace if there's an actual difference
      mv "$temp_file" "$file"
      echo "Updated path in file: $file"
    else
      # No actual changes, discard the temp file
      rm "$temp_file"
      echo "No changes needed in file: $file"
    fi
  fi
done

echo "Path variables updated successfully in classify-unseen-objects directory."