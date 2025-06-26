import wandb
import os
import random

from config import DEFAULT_CONFIG
from pathlib import Path


def log_class_distribution_comparison_before_after_aug(before_aug, after_aug):
    """
    Logs the class distribution before and after augmentation to WandB.
    """
    comparison_data = []
    for cls in sorted(set(before_aug) | set(after_aug)):
        comparison_data.append([
            cls,
            before_aug.get(cls, 0),
            after_aug.get(cls, 0)
        ])
    comp_table = wandb.Table(columns=["Class", "Before Aug", "After Aug"], data=comparison_data)
    wandb.log({"class_distribution_comparison": comp_table})


def log_class_distribution():
    """
    Logs class distribution to Weights & Biases and computes class weights.
    """
    from utils.classes import get_class_distribution
    class_counts = get_class_distribution()

    table = wandb.Table(
        columns=["Class", "Count"],
        data=[[k, v] for k, v in class_counts.items()]
    )

    wandb.log({
        "class_distribution": wandb.plot.bar(table, "Class", "Count", title="Class Distribution"),
        "class_histogram": wandb.plot.histogram(table, "Count", title="Class Distribution Histogram")
    })


def log_class_weights(classes, weights):
    """
    Logs the class weights to WandB.
    """
    weight_table = wandb.Table(columns=["Class", "Weight"],
                               data=[[cls, float(w)] for cls, w in zip(classes, weights)])
    wandb.log({"class_weights_table": weight_table})


def log_sample_images(split='train', samples_per_class=1):
    """
    Logs sample images from the specified split (train/val/test) to Weights & Biases.
    
    Args:
        split (str): Dataset split to pull images from ('train', 'val', 'test').
        samples_per_class (int): Number of images to sample per class.
    """
    print(f"[INFO] Logging {samples_per_class} sample image(s) per class from split: '{split}'")

    dataset_path = Path(DEFAULT_CONFIG['final_dataset_path']) / split
    sample_images = []

    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*'))
            if not images:
                print(f"[WARNING] No images found in: {class_dir}")
                continue

            random.shuffle(images)
            selected_images = images[:samples_per_class]

            for img_path in selected_images:
                sample_images.append(
                    wandb.Image(str(img_path), caption=f"{split}/{class_dir.name}")
                )

    if sample_images:
        wandb.log({f"{split}_dataset_preview": sample_images})
        print(f"[INFO] Logged {len(sample_images)} images to WandB.")
    else:
        print(f"[WARNING] No sample images were logged for split '{split}'.")



