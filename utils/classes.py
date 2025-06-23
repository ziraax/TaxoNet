from collections import defaultdict
from config import CONFIG
from sklearn.utils import compute_class_weight

import os
import numpy as np

from utils.logs import log_class_weights



def get_classes_names():
    """
    Returns a sorted list of class names from the training directory.
    """
    train_path = os.path.join(CONFIG['final_dataset_path'], 'train')
    return sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])


def get_num_classes():
    """
    Returns the number of classes in the dataset.
    """
    return len(get_classes_names())


def get_class_distribution():
    """
    Analyzes how many images exist per class across train/val/test.
    Returns a dictionary mapping class names to image counts.
    """
    class_counts = defaultdict(int)
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(CONFIG['final_dataset_path'], split)
        for class_dir in os.listdir(split_path):
            class_path = os.path.join(split_path, class_dir)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[class_dir] += count
    return class_counts



def get_class_weights(strategy="balanced"):
    """
    Computes class weights using sklearn's compute_class_weight.
    
    Args:
        strategy (str): "balanced" or "uniform". Only "balanced" computes true weights.
    
    Saves:
        CONFIG['class_weights' ] - a list of float weights matching the class order. 
    """ 
    
    class_counts = get_class_distribution()
    classes = sorted(class_counts.keys())

    if strategy == "uniform":
        weights = np.ones(len(classes))
    else:
        y = []
        for cls_idx, cls in enumerate(classes):
            y.extend([cls_idx] * class_counts[cls])
        weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(classes)), y=np.array(y))

    CONFIG['class_weights'] = weights.tolist()
    print(f"[INFO] Computed class weights: {CONFIG['class_weights']}")
    return classes, weights




