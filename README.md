# README

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

### 1. Baseline Configuration

#### 1.1 Model Architecture
**Vanilla U-Net (Baseline)**
- **Depth**: 4 encoder-decoder levels
- **Input Channels**: 4 (RGB + NIR)
- **Output Channels**: 1 (binary segmentation mask)
- **Patch Size**: 128×128 pixels
- **Architecture**: Standard U-Net with skip connections

**Input/Output Specifications:**
```
Input:  image (B, 4, 128, 128)  # Batch, RGBN channels, Height, Width
        label (B, 1, 128, 128)  # Ground truth segmentation mask
Output: pred  (B, 1, 128, 128)  # Predicted segmentation mask
```

#### 1.2 Data Preprocessing Pipeline
1. **Patch Extraction**: Extract 128×128 tiles with stride=64 (50% overlap)
2. **Label Binarization**: Convert mask values > 0 to binary labels
3. **Channel Reordering**: Rearrange to [C, H, W] format
4. **Normalization**: Scale pixel values to [0, 1] range
5. **Optional Padding/Cropping**: Ensure consistent patch sizes
6. **Data Augmentation** (training only):
   - Horizontal/vertical flipping
   - Random rotation
   - Brightness adjustment
   - Contrast adjustment
   - Multiplicative noise
   - Gamma correction
7. **Train/Test Split**: 80% training, 20% testing

#### 1.3 Training Configuration
- **Epochs**: 10 (uniform across all models)
- **Loss Function**: BCE + Dice Loss (weighted combination)
- **Optimizer**: Adam (model-specific learning rates in Table 1)
- **Batch Size**: Model-specific (see Table 1)
- **Evaluation**: Patch-level testing on 128×128 images

#### 1.4 Evaluation Metrics
All metrics computed at pixel level:
- **IoU (Intersection over Union)**: `|A ∩ B| / |A ∪ B|`
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F1-Score**: `2 × (Precision × Recall) / (Precision + Recall)`

### 3. Experimental Protocol

#### 3.1 Training Procedure
1. **Data Loading**: Load patches with specified augmentations
2. **Model Training**: Train for 10 epochs with early stopping

#### 3.2 Evaluation Procedure
1. **Model Loading**: Load best checkpoint for each model
2. **Inference**: Generate predictions on test patches
3. **Post-processing**: Apply refinement methods where specified
4. **Metric Calculation**: Compute IoU, Precision, Recall, F1

#### 3.3 Visualization and Analysis
- Sample predictions vs ground truth
- Success and failure cases

## References
#### U-Net
- **Original Paper**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention* (MICCAI), 234-241.
- **Implementation**: Based on PyTorch U-Net implementation from [pytorch-unet](https://github.com/jvanvugt/pytorch-unet)
- **Code Reference**: [`model/UNet.py`](model/UNet.py)

#### Attention U-Net
- **Original Paper**: Oktay, O., Schlemper, J., Folgoc, L. L., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. *Medical Image Computing and Computer-Assisted Intervention* (MICCAI), 304-312.
- **Paper URL**: https://arxiv.org/pdf/1804.03999.pdf
- **Implementation**: Custom PyTorch implementation with attention gates
- **Code Reference**: [`model/AttentionUNet.py`](model/AttentionUNet.py)

#### ResU-Net
- **Original Paper**: Zhang, Z., Liu, Q., & Wang, Y. (2018). Road Extraction by Deep Residual U-Net. *IEEE Geoscience and Remote Sensing Letters*, 15(5), 749-753.
- **Concept**: Combines residual connections from ResNet with U-Net architecture
- **Implementation**: Custom implementation integrating ResNet blocks into U-Net
- **Code Reference**: [`model/ResUNet.py`](model/ResUNet.py)

#### TransU-Net
- **Original Paper**: Chen, J., Lu, Y., Yu, Q., et al. (2021). TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. *arXiv preprint* arXiv:2102.04306.
- **Paper URL**: https://arxiv.org/abs/2102.04306
- **Implementation**: Hybrid CNN-Transformer architecture with Vision Transformer encoder
- **Code Reference**: [`model/TransUNet.py`](model/TransUNet.py)

#### Hybrid Multi-Task Loss
- **Original Paper**: Rougé, A. D., Stephenson, N. L., Das, A. J., et al. (2023). Dual-Task Learning for Dead Tree Detection and Segmentation with Hybrid Self-Attention U-Nets in Aerial Imagery. *Remote Sensing*, 15(15), 3882.
- **GitHub Repository**: https://github.com/Global-Ecosystem-Health-Observatory/TreeMort
- **Components**:
  - Segmentation Loss: BCE + Dice for mask prediction
  - Centroid Loss: MSE for tree center localization
  - SDT Loss: Smooth L1 + L1 for signed distance transform
- **Code Reference**: [`model/loss.py`](model/loss.py) - `HybridLoss` class

