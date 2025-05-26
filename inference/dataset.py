from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class InferenceDataset(Dataset):
    """
    Custom dataset class for inference.

    Args:
        image_paths (list): List of image file paths.
        img_size (int): Size of the images.    
    """
    def __init__(self, image_paths, img_size):
        self.image_paths = image_paths
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): 
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), img_path
    



