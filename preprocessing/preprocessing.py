import time
import wandb
import yaml
from pathlib import Path
from config import DEFAULT_CONFIG

from preprocessing.augmentor import undersample_overrepresented_classes
from preprocessing.scalebar_removal import remove_scale_bars
from preprocessing.data_organizer import organize_final_structure
from preprocessing.augmentor import apply_class_aware_augmentation

from utils.check_imgs import check_images
from utils.logs import log_class_distribution, log_class_distribution_comparison_before_after_aug, log_class_weights
from utils.classes import get_class_distribution, get_class_weights, get_classes_names
from utils.pytorch_cuda import check_pytorch_cuda


def create_dataset_yaml():
    """
    Creates a dataset.yaml file for YOLOv8.
    """
    class_names = get_classes_names()

    dataset_yaml = {
        'path': DEFAULT_CONFIG['final_dataset_path'],
        'train': 'train',
        'val': 'val',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = Path(DEFAULT_CONFIG['final_dataset_path']) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml.dump(dataset_yaml, default_flow_style=False))

    print(f"[INFO] dataset.yaml created at: {yaml_path}")



def full_preprocessing():
    """
    Full preprocessing pipeline for the original dataset.
    """

    # Check if wandb is initialized
    if wandb.run is None:
        print("[INFO] WandB is not initialized. Please run 'wandb.init()' before calling this function.")
        return
    
    # Check if PyTorch and CUDA are available
    check_pytorch_cuda()

    print(f"[INFO] Starting full preprocessing pipeline...")
    start = time.time()

    # Step 1: Remove scale bars
    remove_scale_bars()
    print(f"Scale bar removal complete. Processed images saved to: {DEFAULT_CONFIG['processed_path']}")

    # Step 2: Check images
    undersample_overrepresented_classes()

    # Check if images are broken
    check_images(DEFAULT_CONFIG['processed_path'], delete=False)

    # Step 3 : Organize final structure
    organize_final_structure()
    print(f"Final structure organized. Processed images saved to: {DEFAULT_CONFIG['final_dataset_path']}")

    # Check images again
    check_images(DEFAULT_CONFIG['final_dataset_path'], delete=False)

    # Step 3.5: Create dataset.yaml
    create_dataset_yaml()
    print(f"dataset.yaml created. Path: {DEFAULT_CONFIG['final_dataset_path']}/dataset.yaml")

    # Get class distribution and log it
    class_distribution_before_aug = get_class_distribution()

    # Step 4 : Augment data
    apply_class_aware_augmentation()
    print(f"[INFO] Data augmentation complete.")
    # Check images after augmentation
    check_images(DEFAULT_CONFIG['final_dataset_path'], delete=False)

    # Get class distribution after augmentation
    class_distribution_after_aug = get_class_distribution()
    log_class_distribution_comparison_before_after_aug(class_distribution_before_aug, class_distribution_after_aug)
    print(f"[INFO] Class distribution comparison logged to Weights & Biases.")

    # Log final class distribution
    log_class_distribution()
    print(f"[INFO] Final class distribution logged to Weights & Biases.")

    # Step 5: Compute class weights
    classes, weights = get_class_weights()
    log_class_weights(classes, weights)

    end = time.time()
    duration = end - start
    print(f"[INFO] Full preprocessing pipeline completed in {duration:.2f} seconds.")
    wandb.log({"full_preprocessing_time_sec": duration})





    



    









