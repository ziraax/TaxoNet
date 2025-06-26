import os
import shutil
import time
import wandb

from pathlib import Path
from sklearn.model_selection import train_test_split
from config import DEFAULT_CONFIG
from datetime import datetime

def organize_final_structure():
    """Convert processed data to proper classification format with proper splits and class filtering."""
    start_time = time.time()
    print("[INFO] Organizing dataset structure...")

    minimum_images_per_class = 10

    processed_path = Path(DEFAULT_CONFIG['processed_path'])
    yolo_path = Path(DEFAULT_CONFIG['final_dataset_path'])

    # Create final dataset directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        (yolo_path / split).mkdir(parents=True, exist_ok=True)

    # Logs directory in root (not under yolo_path)
    root_path = Path(DEFAULT_CONFIG.get("root", "."))
    logs_dir = root_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    skipped_log_path = logs_dir / f"skipped_classes_{timestamp}.txt"

    # Gather valid classes and track skipped ones
    class_images = {}
    skipped_classes = []

    for class_dir in processed_path.glob("*"):
        if class_dir.is_dir():
            class_name = class_dir.name
            images = [img for img in class_dir.glob("*.*") if img.is_file()]
            if len(images) < minimum_images_per_class:
                msg = f"Class '{class_name}' has only {len(images)} image(s) — skipped."
                print(f"[WARNING] {msg}")
                skipped_classes.append(f"{class_name}: {len(images)} image(s)")
                continue
            class_images[class_name] = images

    # Log skipped classes
    if skipped_classes:
        try:
            with open(skipped_log_path, "w") as f:
                f.write(f"Skipped classes (less than {minimum_images_per_class} images):\n")
                f.write("\n".join(skipped_classes))
            print(f"[INFO] Skipped classes logged to: {skipped_log_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write skipped class log: {e}")

    if not class_images:
        raise ValueError(f"[ERROR] No valid classes found with at least {minimum_images_per_class} images. Aborting.")

    # Create full image list
    all_images = [(str(img), class_name) for class_name, imgs in class_images.items() for img in imgs]

    # Train/val/test split
    train, temp = train_test_split(
        all_images,
        test_size=1 - DEFAULT_CONFIG['train_ratio'],
        stratify=[x[1] for x in all_images],
        random_state=DEFAULT_CONFIG['seed']
    )

    val, test = train_test_split(
        temp,
        test_size=DEFAULT_CONFIG['test_ratio'] / (DEFAULT_CONFIG['val_ratio'] + DEFAULT_CONFIG['test_ratio']),
        stratify=[x[1] for x in temp],
        random_state=DEFAULT_CONFIG['seed']
    )

    # Move images into split/class folders
    for split, data in zip(splits, [train, val, test]):
        for img_path, class_name in data:
            src_path = Path(img_path)
            dest_dir = yolo_path / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_path = dest_dir / f"{split}_{src_path.name}"
            if not dest_path.exists():
                if src_path.exists():
                    shutil.copy(src_path, dest_path)
                else:
                    print(f"[WARNING] Missing source file: {src_path}")

    elapsed = time.time() - start_time
    print(f"[INFO] Fianl structure organized in {elapsed:.2f} seconds.")
    wandb.log({"time_organize_yolo_structure_sec": elapsed})
