import os

import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from torch.utils.data import Dataset


class USA_Dataset(Dataset):
    def __init__(self, images, data_root, patch_size=256, stride=64, transform=None, 
                 crop_size=None, is_training=False):
        self.data_root = data_root
        self.rgb_dir = os.path.join(data_root, "RGB_images")
        self.nrg_dir = os.path.join(data_root, "NRG_images")
        self.mask_dir = os.path.join(data_root, "masks")
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.crop_size = crop_size
        self.is_training = is_training
        self.images = images
        self.patch_info = []
        self._compute_patches()

    def _compute_patches(self):
        """Pre-compute patch coordinates for all images"""
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

        # Load and combine RGB + NIR
        rgb = np.array(Image.open(os.path.join(self.rgb_dir, f'RGB_{fname}')))
        nrg = np.array(Image.open(os.path.join(self.nrg_dir, f'NRG_{fname}')))
        image = np.concatenate((rgb, nrg[:, :, 0:1]), axis=2)
        
        # Load mask
        mask = np.array(Image.open(os.path.join(self.mask_dir, f'mask_{fname}')))

        # Extract patches
        image_patch = image[y:y+patch_size, x:x+patch_size, :]
        mask_patch = mask[y:y+patch_size, x:x+patch_size]

        # Pad if necessary
        if image_patch.shape[:2] != (patch_size, patch_size):
            pad_h = max(0, patch_size - image_patch.shape[0])
            pad_w = max(0, patch_size - image_patch.shape[1])
            image_patch = np.pad(image_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Normalize to [0, 1]
        image_patch = image_patch / 255.0
        seg = mask_patch > 0

        # Generate label maps
        centroid = self._generate_centroid_map(seg)
        hybrid = self._generate_hybrid_sdt_map(seg)

        # Convert to tensors [C, H, W]
        image = torch.from_numpy(image_patch).permute(2, 0, 1)
        label = torch.stack([torch.from_numpy(seg), torch.from_numpy(centroid), torch.from_numpy(hybrid)], dim=0)

        if self.crop_size is not None:
            image, label = self._crop_or_pad(image, label, self.crop_size)
        if self.transform is not None and self.is_training:
            image, label = self.transform(image, label)

        return image.float(), label.float()

    def _crop_or_pad(self, image, label, target_size):
        """Crop or pad to target size"""
        c, h, w = image.shape
        th, tw = target_size if isinstance(target_size, (tuple, list)) else (target_size, target_size)
        
        if h > th or w > tw:  # Crop
            start_h, start_w = (h - th) // 2, (w - tw) // 2
            image = image[:, start_h:start_h+th, start_w:start_w+tw]
            label = label[:, start_h:start_h+th, start_w:start_w+tw]
        elif h < th or w < tw:  # Pad
            pad_h, pad_w = (th - h) // 2, (tw - w) // 2
            pad_h_extra, pad_w_extra = th - h - pad_h, tw - w - pad_w
            image = torch.nn.functional.pad(image, (pad_w, pad_w_extra, pad_h, pad_h_extra), mode='reflect')
            label = torch.nn.functional.pad(label, (pad_w, pad_w_extra, pad_h, pad_h_extra), mode='reflect')
        
        return image, label

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