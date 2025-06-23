import torch
import albumentations as A
from pathlib import Path

RAW_DATA_ROOT = Path("DATA/DATA_500_INDIV")
PROCESSED_PATH = Path("DATA/processed_dataset")
FINAL_DATASET_PATH = Path("DATA/final_dataset")

CONFIG = {
    # Project configuration
    "project_name": "YOLOv11Classification500",

    # Raw data structure 
    "years": ["2022", "2023"],
    "types": ["Macro", "Micro"],

    # Path configuration 
    "raw_data_root": str(RAW_DATA_ROOT),
    "processed_path": str(PROCESSED_PATH),
    "final_dataset_path": str(FINAL_DATASET_PATH),

    # Scale bar removal 
    "scalebar_model_path": "models/model_weights/scale_bar_remover/best.pt", # Path to the YOLO model for scale bar removal
    "scalebar_img_size": 416,
    "scalebar_confidence": 0.4,
    "convert_grayscale": True, # convert all images to grayscale since some of them are RGB
    "grayscale_mode": "RGB",

    # Dataset splitting 
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1,
    "seed": 42, # Because its the answer to everything

    # Augmentation parameters
    "augmentation_threshold": 1000,      # Wanted images required per class
    "augmentation_multiplier": 10,       # Maximum copies per original image
    "class_weights": "balanced",         # [This was incorrect - should be "balanced" or None]

    # Augmentation techniques
    "augmentation": {                   # Albumentations pipeline probabilities
        "HorizontalFlip": 0.5,          # 50% chance of horizontal flip
        "VerticalFlip": 0.2,            # 20% chance of vertical flip
        "Rotate90": 0.3,                # 30% chance of 90° rotation
        "BrightnessContrast": 0.4,      # 40% chance of brightness/contrast adjust
        "HueSaturation": 0.3,           # 30% chance of hue/saturation adjust
    },
    

    # YOLOv11 Training
    "model_name": "yolo11l-cls.pt", # By default, use the YOLOv11 classification model
    "img_size": 224, # For YOLOv11
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "num_workers":8, # Number of workers for data loading
}

    # Augmentation pipeline
CONFIG["augmentation_pipeline"] = A.Compose([
    A.HorizontalFlip(p=CONFIG['augmentation']['HorizontalFlip']),
    A.VerticalFlip(p=CONFIG['augmentation']['VerticalFlip']),
    A.RandomRotate90(p=CONFIG['augmentation']['Rotate90']),
    A.RandomBrightnessContrast(p=CONFIG['augmentation']['BrightnessContrast']),
    A.HueSaturationValue(p=CONFIG['augmentation']['HueSaturation']),
])