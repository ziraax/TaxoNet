import os
import time
import torch
import numpy as np
import albumentations as A
import wandb
import math
import hashlib
import random
import concurrent.futures

from PIL import Image
from utils import classes
from config import DEFAULT_CONFIG
from tqdm import tqdm


def undersample_overrepresented_classes():
    """
    Undersamples classes that exceed the augmentation threshold.
    Keeps a random subset of images up to the threshold.
    """
    input_path = DEFAULT_CONFIG['processed_path']
    max_per_class = DEFAULT_CONFIG['augmentation_threshold']

    for class_name in os.listdir(input_path):
        class_dir = os.path.join(input_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(image_files) > max_per_class:
            print(f"[INFO] Undersampling {class_name}: {len(image_files)} -> {max_per_class}")
            keep = set(random.sample(image_files, max_per_class))
            for img_file in image_files:
                if img_file not in keep:
                    os.remove(os.path.join(class_dir, img_file))

    print("[INFO] Undersampling complete.")


def apply_class_aware_augmentation():
    """
    Augments underrepresented classes to reach the augmentation threshold.
    Ensures not to exceed augmentation_multiplier per image.
    Uses multiprocessing for efficiency.
    """
    train_path = os.path.join(DEFAULT_CONFIG['final_dataset_path'], 'train')
    class_counts = classes()
    tasks = []

    for class_name, count in class_counts.items():
        if count < DEFAULT_CONFIG['augmentation_threshold']:
            class_dir = os.path.join(train_path, class_name)
            original_images = sorted([
                f for f in os.listdir(class_dir)
                if not f.startswith('aug_') and f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            needed = DEFAULT_CONFIG['augmentation_threshold'] - count
            num_originals = len(original_images)

            if num_originals == 0:
                print(f"[WARNING] Skipping {class_name} — no original images")
                continue

            per_image = max(1, math.ceil(needed / num_originals))
            if per_image > DEFAULT_CONFIG['augmentation_multiplier']:
                print(f"[WARN] Class '{class_name}' cannot reach {DEFAULT_CONFIG['augmentation_threshold']} "
                      f"with only {num_originals} originals and multiplier={DEFAULT_CONFIG['augmentation_multiplier']}.")
                per_image = DEFAULT_CONFIG['augmentation_multiplier']

            print(f"[INFO] Class {class_name}: {count} -> {DEFAULT_CONFIG['augmentation_threshold']} | "
                  f"{per_image} augmentations per image")

            for idx, img_name in enumerate(original_images):
                img_path = os.path.join(class_dir, img_name)
                tasks.append((img_path, class_dir, idx, per_image, img_name))

    # Parallel processing
    total_augmented = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(augment_image_task, tasks), total=len(tasks), desc="Augmenting images"):
            total_augmented += result

    print(f"[INFO] Augmentation complete. Total augmented images: {total_augmented}")
    wandb.log({"total_augmented_images": total_augmented})


def augment_image_task(args):
    """
    Subprocess task: Applies multiple augmentations to a single image and saves them.
    Creates the Albumentations pipeline locally to avoid pickling issues.
    """
    img_path, class_dir, idx, per_image, img_name = args

    def build_pipeline():
        return A.Compose([
            A.HorizontalFlip(p=DEFAULT_CONFIG['augmentation']['HorizontalFlip']),
            A.VerticalFlip(p=DEFAULT_CONFIG['augmentation']['VerticalFlip']),
            A.RandomRotate90(p=DEFAULT_CONFIG['augmentation']['Rotate90']),
            A.RandomBrightnessContrast(p=DEFAULT_CONFIG['augmentation']['BrightnessContrast']),
            A.HueSaturationValue(p=DEFAULT_CONFIG['augmentation']['HueSaturation']),
        ])

    try:
        image = np.array(Image.open(img_path).convert("RGB"))
    except Exception as e:
        print(f"[ERROR] Could not load image {img_path}: {str(e)}")
        return 0

    successful = 0
    pipeline = build_pipeline()

    for copy_num in range(per_image):
        seed = int(hashlib.sha256(img_name.encode()).hexdigest()[:8], 16) + copy_num
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        try:
            augmented = pipeline(image=image)['image']
            aug_img = Image.fromarray(augmented)
            aug_path = os.path.join(class_dir, f"aug_{idx}_{copy_num}_{img_name}")
            aug_img.save(aug_path)
            successful += 1
        except Exception as e:
            print(f"[ERROR] Augmentation failed for {img_path}: {str(e)}")

    return successful
