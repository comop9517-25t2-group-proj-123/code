# README

## Task

## Dataset
aerial multispectral image samples from the US, The data are manually annotated by our collaborator group of forest health experts.
- 444 annotated scenes available with relatively smaller dimensions around 300 × 300 pixels.
- Included scenes span multiple states with a ground resolution of 60 cm. 
- The image samples have four-band data, including near-infrared (NIR) and RGB channels .png format.
- consists of annotations for standing dead trees

### Data preprocess (dataset.py)
1. Patchify: extract image tiles from the train split with a size of 256 × 256 using a stride size of 64 for the USA data.
2. stack [mask, centroid, hybrid] as label
3. reorder to [C, H, W]
4. normalize to [0, 1]
5. optional crop / pad
6. Augmentation (for trainning data): flip, rotation, brightness, contrast, multiplicative_noise, gamma

### dataloader.py
- train, test, val split
val-size = 0.2
test-size = 0.1

## Model
- Input
image: (B, 4, 256, 256)
label: (B, 3, 256, 256)
- Output
label: (B, 3, 256, 256)
- Model
unet

## Loss
Segmentation loss: A combination of weighted Binary Cross-Entropy (BCE) loss and Dice loss drives the accurate separation of dead tree canopies from the background:

L_seg = L_BCE + λ_Dice * L_Dice

## Evaluation Metric
mIoU

## Configuration
- dataset cfg
    - patch_size
    - stride step
- dataloader cfg
    - batch_size
- model
    - in_channels 3/4
    - out_channels 1/3
    - depth
- trainer
    - lr rate
    - epoch

## Experiment
### Model-wise
- model size and data size
```
model_configs = [
    {'depth': 3, 'patch_size': 128},
    {'depth': 4, 'patch_size': 128},
    {'depth': 4, 'patch_size': 256},
]
```
- different model
### Postprocessing-wise
- Which postprocessing method contributes most to semantic segmentation performance?

- What's the optimal combination of methods?

- How do different parameter settings affect performance?
``` python
threshold_study = {
    'seg_thresh': [0.3, 0.5, 0.7],  # Segmentation threshold
    'hyb_thresh': [0.3, 0.5, 0.7],  # Hybrid filtering threshold
    'min_area': [20, 50, 100],      # Minimum area filtering
}
```
###