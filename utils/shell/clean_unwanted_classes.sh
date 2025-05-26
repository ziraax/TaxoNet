#!/bin/bash

# Base path
BASE_DIR="DATA/DATA_500_INDIV"

# Years and types
YEARS=("2022" "2023")
TYPES=("Macro" "Micro")

# Classes to remove
CLASSES_TO_REMOVE=(
  "Duplicate"
  "not-living"
  "Nymphes"
  "othertocheck"
  "t001" "t002" "t003" "t004" "t005"
  "other_living"
  "badfocus_other"
  "tempgrey"
  "Harpacticoida"
  "Willea"
  "Brachionus urceolaris"
  "Closteriopsis"
  "Gastropoda_Mollusca"
  "Graptoleberis"
  "Tardigrada sp_"
  "Cyanobacteria_Bacteria"
  "Metazoa X"
  "Vorticella"
  "Viridiplantae"
)

echo "=== Deleting unwanted class folders ==="
for year in "${YEARS[@]}"; do
  for type in "${TYPES[@]}"; do
    DIR="$BASE_DIR/$year/$type"
    for class in "${CLASSES_TO_REMOVE[@]}"; do
      CLASS_DIR="$DIR/$class"
      if [ -d "$CLASS_DIR" ]; then
        echo "[INFO] Found and removing: $CLASS_DIR"
        rm -rf "$CLASS_DIR"
      else
        echo "[SKIP] Folder not found: $CLASS_DIR"
      fi
    done
  done
done

echo ""
echo "=== Checking remaining class folders for low image count ==="
for year in "${YEARS[@]}"; do
  for type in "${TYPES[@]}"; do
    DIR="$BASE_DIR/$year/$type"
    echo "-- Checking: $DIR"
    for class_dir in "$DIR"/*; do
      if [ -d "$class_dir" ]; then
        count=$(find "$class_dir" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l)
        if [ "$count" -le 10 ]; then
          echo "[WARNING] '$class_dir' has only $count image(s)."
        fi
      fi
    done
  done
done
