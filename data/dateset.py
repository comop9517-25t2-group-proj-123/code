import os

import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from torch.utils.data import Dataset


class USA_Dataset(Dataset):
    def __init__(self, images, cfg, transform=None, is_training=False):
        self.cfg = cfg
        ds_cfg = cfg['dataset']
        self.data_root = ds_cfg['data_root']
        self.rgb_dir = os.path.join(self.data_root, "RGB_images")
        self.nrg_dir = os.path.join(self.data_root, "NRG_images")
        self.mask_dir = os.path.join(self.data_root, "masks")
        self.patch_size = ds_cfg.get('patch_size', 256)
        self.stride = ds_cfg.get('stride', 64)
        self.nrg = ds_cfg.get('nrg', True)
        self.transform = transform
        self.is_training = is_training
        self.images = images
        self.patch_info = []
        self._compute_patches()

    def _compute_patches(self):
        for img_name in self.images:
            rgb_path = os.path.join(self.rgb_dir, f'RGB_{img_name}')
            if os.path.exists(rgb_path):
                img = Image.open(rgb_path)
                height, width = img.size[1], img.size[0]
                for y in range(0, height - self.patch_size + 1, self.stride):
                    for x in range(0, width - self.patch_size + 1, self.stride):
                        self.patch_info.append({
                            'image_name': img_name, 'x': x, 'y': y, 'patch_size': self.patch_size
                        })

    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        patch_data = self.patch_info[idx]
        fname, x, y, patch_size = patch_data['image_name'], patch_data['x'], patch_data['y'], patch_data['patch_size']

        rgb = np.array(Image.open(os.path.join(self.rgb_dir, f'RGB_{fname}')))
        nrg = np.array(Image.open(os.path.join(self.nrg_dir, f'NRG_{fname}')))
        mask = np.array(Image.open(os.path.join(self.mask_dir, f'mask_{fname}')))
        
        if self.nrg:
            image = np.concatenate((rgb, nrg[:, :, 0:1]), axis=2)
        else:
            image = rgb

        image = image[y:y+patch_size, x:x+patch_size, :]
        mask = mask[y:y+patch_size, x:x+patch_size]

        image = image / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1) # [C, H, W]
        mask_binary = mask > 0  # Keep as numpy array initially

        if self.cfg['trainer']['hybrid_loss']:
            centroid = self._generate_centroid_map(mask_binary)
            hybrid = self._generate_hybrid_sdt_map(mask_binary)
            # Stack all three channels
            label = torch.stack([
                torch.from_numpy(mask_binary.astype(np.float32)),
                torch.from_numpy(centroid.astype(np.float32)), 
                torch.from_numpy(hybrid.astype(np.float32))
            ], dim=0)  # [3, H, W]
        else:
            label = torch.from_numpy(mask_binary.astype(np.float32)).unsqueeze(0)  # [1, H, W]
    
        if self.transform is not None and self.is_training:
            image, label = self.transform(image, label)

        return image.float(), label.float()

    def _generate_centroid_map(self, seg):
        """Generate centroid map from segmentation mask"""
        centroid = np.zeros_like(seg)
        labeled, n_trees = ndi.label(seg)
        if n_trees > 0:
            centers = ndi.center_of_mass(seg, labeled, range(1, n_trees+1))
            for center in centers:
                if len(center) == 2:
                    y_c, x_c = int(center[0]), int(center[1])
                    if 0 <= y_c < centroid.shape[0] and 0 <= x_c < centroid.shape[1]:
                        centroid[y_c, x_c] = 1.0
        return ndi.gaussian_filter(centroid, sigma=3)

    def _generate_hybrid_sdt_map(self, seg):
        """Generate hybrid signed distance transform map"""
        sdt = ndi.distance_transform_edt(seg) / 32.0
        inv_sdt = ndi.distance_transform_edt(1 - seg) / 32.0
        hybrid = sdt - inv_sdt
        hybrid[seg == 1] = sdt[seg == 1]
        hybrid[np.logical_xor(seg, ndi.binary_erosion(seg))] = -1
        return hybrid