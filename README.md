# README
## Task
The goal of this group project is to develop and compare different computer vision methods
for segmenting standing dead trees in aerial images of forests.
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
test on patch 128*128 image (table 2)
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
| Original Image Size | Around 300 × 300 pixels |
| Patch Size | 128 × 128 pixels |
| Patch Stride | 64 pixels |
| Number of Patches | 8,880 (20 patches per scene) |
| Patch Overlap (%) | ~50% overlapping pixels |
| Augmentation | flip, rotation, brightness, contrast, multiplicative_noise, gamma |
| Train/Test Split | 80% / 20% (355 / 89 scenes) |

**Table 1 for Model HYPERPARAMETER VALUES**
| Model | Params (M) | FLOPs (G) | Batch Size | Learning Rate | Optimizer |
|-------|------------|-----------|------------|---------------|-----------|
| U-Net | 7.7 | 10.5 | 16 | 1e-4 | Adam |
| U-Net (w/o NIR) | 7.7 | 10.4 | 16 | 1e-4 | Adam |
| Attention U-Net | 34.9 | 16.7 | 16 | 1e-4 | Adam |
| ResU-Net |  36.0 | 17.9 | 16 | 5e-5 | Adam |
| TransU-Net | 140.8 | 12.0 | 16 | 5e-5 | Adam |


| ResNet-50 (Linear) | 25.6 | 4.1 | 32 | 1e-3 | SGD |
| ViT-Base (Linear) | 86.4 | 17.6 | 16 | 1e-3 | AdamW |
| CLIP-Base (Linear) | 86.4 | 17.6 | 16 | 5e-4 | AdamW |
| DINOv2-Base (Linear) | 86.4 | 17.6 | 16 | 1e-3 | AdamW |

**Tabel 2 for different model results**
| Model | Pretrained | mIoU | F1 | Precision | Recall |
|-------|------------|------|----|-----------| -------|
| U-Net | No | 0.395 | 0.509 | 0.539 | 0.557 |
| U-Net (w/o NIR) | No | 0.381 | 0.495 | 0.487 | 0.585 |
| Attention U-Net | No | 0.394 | 0.503 | 0.591 | 0.506 |
| ResU-Net | No | 0.387 | 0.502 | 0.566 | 0.528 |
| TransU-Net | No | 0.381 | 0.492 | 0.590 | 0.491 |


| ResNet-50 (Linear) | ImageNet | 
| ViT-Base (Linear) | ImageNet | 
| CLIP-Base (Linear) | CLIP | 
| DINOv2-Base (Linear) | Self-supervised | 

**Tabel 3 for Postprocessing**
| Method | Threshold | Min Area | mIoU | mIoU Improvement |
|--------|-------------------|------|------------------|---------------------|
| None (Baseline) | - | - | 0.395 | - |
| Segmentation Refinement | 0.5 | 20 | 0.400 | +0.005 |
| Segmentation Refinement | 0.7 | 20 | 0.401 | +0.006 |
| Segmentation Refinement | 0.5 | 50 | 0.389 | -0.006 |

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