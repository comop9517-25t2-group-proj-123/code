# README
## Task
The goal of this group project is to develop and compare different computer vision methods
for segmenting standing dead trees in aerial images of forests.
The goal of this group project is to develop and compare different computer vision methods
for segmenting standing dead trees in aerial images of forests.

## Dataset Desciption
aerial multispectral image samples from the US, The data are manually annotated by our collaborator group of forest health experts.
- 444 annotated scenes available with relatively smaller dimensions around 300 × 300 pixels.
- Included scenes span multiple states with a ground resolution of 60 cm. 
- The image samples have four-band data, including near-infrared (NIR) and RGB channels .png format.
- consists of annotations for standing dead trees

## Experiment Design
### Baseline Setup
#### Baseline model (in cofig)
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


### Visualization
- successful segmentations
- failed segmentations

## References
Libraries or code obtained from other sources should be clearly described

## Run code (Demonstration)
- Include proper documentation about how to run the code
- add tutorial jupyter notebooks for visualization
    - the presentation must include a demonstration of the methods/software in action