import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data.dateset import USA_Dataset
from data.augment import Augmentations


def get_image_filenames(data_dir):
    """Get all image filenames from the USA_segmentation dataset"""
    rgb_dir = os.path.join(data_dir, "RGB_images")
    image_files = []
    
    if os.path.exists(rgb_dir):
        for file in os.listdir(rgb_dir):
            if file.startswith("RGB_") and file.endswith(".png"):
                # Extract the base filename (remove RGB_ prefix)
                base_name = file[4:]  # Remove "RGB_" prefix
                image_files.append(base_name)
    
    return sorted(image_files)


def stratify_images_by_filename(image_files, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """Split images into train/val/test sets based on filename patterns"""
    # First split: separate test set
    train_val_files, test_files = train_test_split(
        image_files, 
        test_size=test_ratio, 
        random_state=random_state
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_ratio / (1 - test_ratio)  # Adjust val ratio for remaining data
    train_files, val_files = train_test_split(
        train_val_files, 
        test_size=val_size_adjusted, 
        random_state=random_state
    )
    
    return train_files, val_files, test_files


def prepare_dataloaders(data_root, val_ratio=0.2, test_ratio=0.1, patch_size=256, stride=64,
                       train_batch_size=8, val_batch_size=16, test_batch_size=16, num_workers=4):
    """Complete pipeline: get images, split data, create datasets and dataloaders"""
    # Get image files and split
    image_files = get_image_filenames(data_root)
    if not image_files:
        raise ValueError(f"No images found in {data_root}")
    
    train_files, val_files, test_files = stratify_images_by_filename(
        image_files, val_ratio=val_ratio, test_ratio=test_ratio
    )
    
    # Create datasets
    train_dataset = USA_Dataset(train_files, data_root, patch_size, stride, Augmentations(), is_training=True)
    val_dataset = USA_Dataset(val_files, data_root, patch_size, stride, is_training=False)
    test_dataset = USA_Dataset(test_files, data_root, patch_size, stride, is_training=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                           num_workers=num_workers, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False, pin_memory=True)
    
    return train_loader, val_loader, test_loader
