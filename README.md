# README
## Task
The goal of this group project is to develop and compare different computer vision methods
for segmenting standing dead trees in aerial images of forests.
The goal of this group project is to develop and compare different computer vision methods
for segmenting standing dead trees in aerial images of forests.

## Getting Started
### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (cuda 12.6 for experiment)

**Create conda environment from YAML:**
```bash
conda env create -f environment.yml
conda activate comp9517
```

### Dataset Setup

1. **Download the dataset:**
   - Download the USA_segmentation dataset
   - Extract to `data/datasets/USA_segmentation/`

2. **Expected directory structure:**
```
data/datasets/USA_segmentation/
├── RGB_images/          # RGB aerial images
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
├── NRG_images/          # Near-infrared, Red, Green images  
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
└── masks/               # Binary segmentation masks
    ├── image_001.tif
    ├── image_002.tif
    └── ...
```

### Running the Code

#### Quick Start
```bash
python main.py
```

#### Configuration Options
Edit `config/config.py` to modify training parameters:
```python
# Basic configuration example
cfg = {
    'dataset': {
        'patch_size': 128,        # 64, 128, 256
        'stride': 64,             # Overlap between patches
        'nrg': True,              # Use NIR channel
    },
    'model': {
        'name': 'UNet',           # UNet, AttentionUNet, ResUNet, TransUNet
        'in_channels': 4,         # 3 (RGB) or 4 (RGB+NIR)
        'n_classes': 1,           # 1 (segmetation mask) or 3 (multi-task)
        'depth': 4,               # Model depth
    },
    'trainer': {
        'learning_rate': 1e-3,    
        'epochs': 10,
        'hybrid_loss': False,     # True for multi-task learning
    },
    'dataloader': {
        'train_batch_size': 16,   # Adjust based on GPU memory
        'num_workers': 4,
    }
}
```
#### Available Models
- **UNet**: Vanilla U-Net architecture
- **AttentionUNet**: U-Net with attention gates
- **ResUNet**: U-Net with residual connections
- **TransUNet**: Transformer-based U-Net

#### Loss Functions
- **BCEDiceLoss** (default): Binary Cross-Entropy + Dice Loss
- **HybridLoss**: Multi-task loss for segmentation + centroid detection + SDT

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
- Unet (w/o NIG channel)
- Attention Unet
- ResUNet
- TransUnet

### BCE loss vs Hybrid loss

### Postprocessing comparison
**Initial Segmentation Refinement** (noise removal)

### Visualization
- successful segmentations
- failed segmentations

## References
Libraries or code obtained from other sources should be clearly described