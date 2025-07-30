import random

import torch


def random_flip(image, label):
    """Apply random horizontal and vertical flips"""
    if random.random() > 0.5:
        image, label = torch.flip(image, [2]), torch.flip(label, [2])
    if random.random() > 0.5:
        image, label = torch.flip(image, [1]), torch.flip(label, [1])
    return image, label


def random_rotation(image, label):
    """Apply random 90-degree rotations"""
    k = random.randint(0, 3)
    return torch.rot90(image, k, [1, 2]), torch.rot90(label, k, [1, 2])


def random_brightness(image, label):
    """Apply random brightness adjustment to image only"""
    factor = 1.0 + random.uniform(-0.2, 0.2)
    return torch.clamp(image * factor, 0, 1), label


def random_contrast(image, label):
    """Apply random contrast adjustment to image only"""
    factor = 1.0 + random.uniform(-0.2, 0.2)
    mean = torch.mean(image, dim=(1, 2), keepdim=True)
    return torch.clamp((image - mean) * factor + mean, 0, 1), label


def random_multiplicative_noise(image, label):
    """Apply random multiplicative noise to image only"""
    noise = torch.rand_like(image) * 0.2 + 0.9
    return torch.clamp(image * noise, 0, 1), label


def random_gamma(image, label):
    """Apply random gamma correction to image only"""
    gamma = random.uniform(0.8, 1.2)
    return torch.clamp(image ** gamma, 0, 1), label


class Augmentations:
    """Applies all augmentations in sequence"""
    
    def __call__(self, image, label):
        for aug_fn in [random_flip, random_rotation, random_brightness, 
                       random_contrast, random_multiplicative_noise, random_gamma]:
            image, label = aug_fn(image, label)
        return image, label
