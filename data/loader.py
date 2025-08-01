import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data.dateset import USA_Dataset
from data.augment import Augmentations

def get_image_filenames(data_dir):
    rgb_dir = os.path.join(data_dir, "RGB_images")
    image_files = []
    if os.path.exists(rgb_dir):
        for file in os.listdir(rgb_dir):
            if file.startswith("RGB_") and file.endswith(".png"):
                base_name = file[4:]
                image_files.append(base_name)
    return sorted(image_files)

def prepare_dataloaders(cfg):
    data_root = cfg['dataset']['data_root']
    test_ratio = cfg['dataset'].get('test_ratio', 0.2)
    train_batch_size = cfg['dataloader'].get('train_batch_size', 8)
    test_batch_size = cfg['dataloader'].get('test_batch_size', 16)
    num_workers = cfg['dataloader'].get('num_workers', 4)

    image_files = get_image_filenames(data_root)
    train_files, test_files = train_test_split(
        image_files, 
        test_size=test_ratio, 
        random_state=42
    )

    train_dataset = USA_Dataset(
        train_files, cfg, transform=Augmentations(), is_training=True
    )
    test_dataset = USA_Dataset(
        test_files, cfg, is_training=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )
    return train_loader, test_loader