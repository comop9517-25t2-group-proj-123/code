# README
## Task
The goal of this group project is to develop and compare different computer vision methods
for segmenting standing dead trees in aerial images of forests.

## Dataset
aerial multispectral image samples from the US, The data are manually annotated by our collaborator group of forest health experts.
- 444 annotated scenes available with relatively smaller dimensions around 300 × 300 pixels.
- Included scenes span multiple states with a ground resolution of 60 cm. 
- The image samples have four-band data, including near-infrared (NIR) and RGB channels .png format.
- consists of annotations for standing dead trees

## Experiment Design
### Baseline Setup
#### Baseline model
Vannila Unet
- Architecture: Depth=4
- Input
    - image: (B, 4, 128, 128)
    - label: (B, 1, 128, 128)
- Output
    - label: (B, 1, 128, 128)

#### Data preprocess
1. Patchify: extract image tiles with a size of 128 × 128 using a stride size of 64
2. set mask > 0 as label
3. reorder to [C, H, W]
4. normalize to [0, 1]
5. optional crop / pad
6. Augmentation (for trainning data): flip, rotation, brightness, contrast, multiplicative_noise, gamma
- The dataset is split into 80% for training, and 20% for testing

#### Training configuration
- a uniform training schedule of 10 epochs across models
- learning rate, batch size, and optimizer used for each model specify in table 1
- Loss: Binary Cross-Entropy + Dice Loss (weighted combination)

#### Evaluation Metrics
test on full 300×300 image (table 2)
- Pixel IoU (Intersection over Union)
- Precision
- Recall
- F1-Score

### Model Comparison
- Machine learning
- Unet (w/o NIG channel)
- Attention Unet
- ResUNet
- TransUnet
- U-Net (EfficientNet-B7)
- UNet++ (EfficientNet-B7)

#### Pretrained models (linear probing)
ResNet-50 ResNet-50 ViT-Base ViT-Base CLIP-Base DINOv2-Base
- Minimal Linear Head
- Apply heavy augmentation

### Postprocessing comparison
1. **None** (baseline threshold only)
2. **Initial Segmentation Refinement** (noise removal)
3. **Hybrid Filtering** (boundary refinement)
4. **Combined** (method 2 + 3)

## Results (Example, TODO)
### Quantitative results
**Tabel 0 for Dataset processing**
| Dataset |  |
|--------|-------|
| Total Images | 444 scenes |
| Spectral Bands | Red, Green, Blue, Near-Infrared (4 channels) |
| Original Image Size | 300 × 300 pixels |
| Number of Patches | 8,880 (20 patches per scene) |
| Patch Size | 128 × 128 pixels |
| Patch Stride | 64 pixels |
| Patch Overlap (%) | ~50% overlapping pixels |
| Augmentation | flip, rotation, brightness, contrast, multiplicative_noise, gamma |
| Train/Test Split | 80% / 20% (355 / 89 scenes) |
**Table 1 for Model HYPERPARAMETER VALUES**
| Model | Params (M) | FLOPs (G) | Batch Size | Learning Rate | Optimizer |
|-------|------------|-----------|------------|---------------|-----------|
| U-Net | 31.4 | 12.8 | 16 | 1e-4 | Adam |
| U-Net (w/o NIR) | 28.9 | 11.2 | 16 | 1e-4 | Adam |
| Attention U-Net | 34.9 | 15.3 | 12 | 8e-5 | Adam |
| ResU-Net | 42.1 | 18.7 | 12 | 1e-4 | AdamW |
| TransU-Net | 105.3 | 45.2 | 8 | 5e-5 | AdamW |
| U-Net (EfficientNet-B7) | 66.7 | 28.4 | 8 | 5e-5 | AdamW |
| UNet++ (EfficientNet-B7) | 89.2 | 35.1 | 6 | 3e-5 | AdamW |
| ResNet-50 (Linear) | 25.6 | 4.1 | 32 | 1e-3 | SGD |
| ViT-Base (Linear) | 86.4 | 17.6 | 16 | 1e-3 | AdamW |
| CLIP-Base (Linear) | 86.4 | 17.6 | 16 | 5e-4 | AdamW |
| DINOv2-Base (Linear) | 86.4 | 17.6 | 16 | 1e-3 | AdamW |

**Tabel 2 for different model results**
| Model | Pretrained | mIoU | F1 | Precision | Recall |
|-------|------------|------|----|-----------| -------|
| U-Net | No | 0.742 | 0.851 | 0.834 | 0.869 |
| U-Net (w/o NIR) | No | 0.718 | 0.836 | 0.821 | 0.851 |
| Attention U-Net | No | 0.751 | 0.857 | 0.845 | 0.870 |
| ResU-Net | No | 0.758 | 0.862 | 0.851 | 0.874 |
| TransU-Net | ImageNet | 0.769 | 0.870 | 0.863 | 0.877 |
| U-Net (EfficientNet-B7) | ImageNet | 0.773 | 0.873 | 0.869 | 0.878 |
| UNet++ (EfficientNet-B7) | ImageNet | 0.781 | 0.878 | 0.875 | 0.881 |
| ResNet-50 (Linear) | ImageNet | 0.685 | 0.813 | 0.798 | 0.829 |
| ViT-Base (Linear) | ImageNet | 0.701 | 0.825 | 0.812 | 0.838 |
| CLIP-Base (Linear) | CLIP | 0.694 | 0.820 | 0.805 | 0.835 |
| DINOv2-Base (Linear) | Self-supervised | 0.708 | 0.829 | 0.816 | 0.842 |

**Tabel 3 for Postprocessing**
| Method | Setting Parameters | mIoU | mIoU Improvement | Processing Time (ms) |
|--------|-------------------|------|------------------|---------------------|
| None (Baseline) | threshold=0.5 | 0.742 | - | 0.1 |
| Initial Segmentation Refinement | threshold=0.5, min_area=50 | 0.756 | +0.014 | 2.3 |
| Hybrid Filtering | hyb_thresh=0.5 | 0.748 | +0.006 | 4.7 |
| Combined | threshold=0.5, min_area=50, hyb_thresh=0.5 | 0.761 | +0.019 | 6.8 |

### Visualization
- successful segmentations
- failed segmentations

## Discussion
Give some
explanation why you believe your methods failed in these cases. Also, if one of your methods
clearly works better than the other(s), discuss possible reasons why. Finally, discuss some
potential directions for future research to further improve segmentation performance.

## References
Libraries or code obtained from other sources should be clearly described

## Run code (Demonstration)
- Include proper documentation about how to run the code
- add tutorial jupyter notebooks for visualization
    - the presentation must include a demonstration of the methods/software in action