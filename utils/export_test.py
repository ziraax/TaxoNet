import os
import shutil
from pathlib import Path
import json
import zipfile
from datetime import datetime

SOURCE_DIR = Path("DATA/yolo_dataset/test")
EXPORT_DIR = Path("exports/test_images")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

image_to_class = {}

# Copy images and build image_to_class mapping
for class_dir in SOURCE_DIR.iterdir():
    if class_dir.is_dir():
        class_name = class_dir.name
        for image_path in class_dir.glob("*.*"):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                dest_path = EXPORT_DIR / image_path.name
                shutil.copy(image_path, dest_path)
                image_to_class[image_path.name] = class_name

# Save image_to_class mapping to JSON
json_path = EXPORT_DIR / "image_to_class.json"
with open(json_path, "w") as f:
    json.dump(image_to_class, f, indent=4)

# Write class_names.txt
class_names = sorted({v for v in image_to_class.values()})
class_names_path = EXPORT_DIR / "class_names.txt"
with open(class_names_path, "w") as f:
    for cls in class_names:
        f.write(cls + "\n")

print(f"Exported {len(image_to_class)} images to {EXPORT_DIR}")
print(f"Saved image-to-class map to {json_path}")
print(f"Saved class names to {class_names_path}")

# Create timestamped zip file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_name = f"test_images_export_{timestamp}.zip"
zip_path = EXPORT_DIR.parent / zip_name

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file_path in EXPORT_DIR.iterdir():
        zipf.write(file_path, arcname=file_path.name)

# Print size of zip file
size_mb = zip_path.stat().st_size / (1024 * 1024)
print(f"Created zip archive: {zip_path} ({size_mb:.2f} MB)")
