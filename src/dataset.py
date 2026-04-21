"""
Dataset loading and preprocessing for FER2013.
Handles data augmentation, normalization, and batch creation.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config


class FER2013Dataset(Dataset):
    """
    Custom Dataset for FER2013 facial emotion recognition dataset.
    Expects directory structure:
    fer2013/
        angry/
        disgust/
        fear/
        happy/
        sad/
        surprise/
        neutral/
    """
    
    def __init__(self, root_dir=config.FER2013_DIR, split='train', transform=None, augment=False):
        """
        Args:
            root_dir (str): Path to FER2013 directory with emotion subdirectories
            split (str): 'train', 'val', or 'test'
            transform (torchvision.transforms.Compose): Transforms to apply
            augment (bool): Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.split = split
        self.augment = augment
        self.images = []
        self.labels = []
        
        # Load all images and their labels
        for emotion_idx, emotion_name in config.EMOTIONS.items():
            emotion_dir = os.path.join(root_dir, emotion_name)
            
            if not os.path.exists(emotion_dir):
                print(f"Warning: {emotion_dir} not found. Make sure FER2013 is downloaded.")
                continue
            
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # Split data
            np.random.seed(config.RANDOM_SEED)
            np.random.shuffle(image_files)
            
            total = len(image_files)
            train_end = int(total * 0.7)
            val_end = int(total * 0.85)
            
            if split == 'train':
                image_files = image_files[:train_end]
            elif split == 'val':
                image_files = image_files[train_end:val_end]
            elif split == 'test':
                image_files = image_files[val_end:]
            
            for img_file in image_files:
                self.images.append(os.path.join(emotion_dir, img_file))
                self.labels.append(emotion_idx)
        
        self.transform = transform or self._get_default_transform()
    
    def _get_default_transform(self):
        """Get default transforms based on split and augmentation settings."""
        if self.split == 'train' and self.augment:
            return transforms.Compose([
                transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
                transforms.RandomRotation(config.AUGMENTATION_CONFIG['rotation']),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=config.AUGMENTATION_CONFIG['brightness'],
                    contrast=config.AUGMENTATION_CONFIG['contrast']
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=config.AUGMENTATION_CONFIG['translate']
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD),
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image (convert to RGB if grayscale)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(batch_size=32, num_workers=4):
    """
    Create train, validation, and test data loaders.
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
    
    Returns:
        dict: Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    
    # Create datasets
    train_dataset = FER2013Dataset(split='train', augment=True)
    val_dataset = FER2013Dataset(split='val', augment=False)
    test_dataset = FER2013Dataset(split='test', augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Test dataset loading
    print("Testing FER2013Dataset...")
    dataset = FER2013Dataset(split='train', augment=True)
    print(f"Train dataset size: {len(dataset)}")
    
    # Test data loader
    loaders = get_data_loaders(batch_size=32)
    train_loader = loaders['train']
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]}")
    print("Dataset test passed!")
