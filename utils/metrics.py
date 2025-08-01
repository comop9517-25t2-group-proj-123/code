import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist


def pixel_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    
    return inter / union
