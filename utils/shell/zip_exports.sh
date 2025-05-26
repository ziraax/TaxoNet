#!/bin/bash

EXPORTS_DIR="exports"

# Loop through each subfolder in exports/
for dir in "$EXPORTS_DIR"/*/; do
    # Remove trailing slash and get base folder name
    folder_name=$(basename "$dir")
    zip_file="${EXPORTS_DIR}/${folder_name}.zip"

    echo "Zipping $folder_name to $zip_file..."
    zip -r -q "$zip_file" "$dir"

    # Get file size in human-readable format
    if [ -f "$zip_file" ]; then
        size=$(du -h "$zip_file" | cut -f1)
        echo "Created $zip_file (${size})"
    else
        echo "Failed to create $zip_file"
    fi
done

echo "All subfolders zipped and sizes listed."
