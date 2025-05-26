import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import CONFIG


def get_default_transforms():
    """
    Returns default transforms: Resize, ToTensor, Normalize
    """
    return transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet values
                             std=[0.229, 0.224, 0.225])
    ])


def get_train_dataloader(config):
    """
    Returns a DataLoader for the training dataset.
    """
    train_dir = os.path.join(config['yolo_dataset_path'], 'train')
    transform = get_default_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True)
    return train_loader


def get_test_dataloader(config):
    """
    Returns a DataLoader for the test dataset.
    """
    test_dir = os.path.join(config['yolo_dataset_path'], 'test')
    transform = get_default_transforms()

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers'],
                             pin_memory=True)
    return test_loader


def get_val_dataloader(config):
    """
    Returns a DataLoader for the validation dataset.
    """
    val_dir = os.path.join(config['yolo_dataset_path'], 'val')
    transform = get_default_transforms()

    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True)
    return val_loader
