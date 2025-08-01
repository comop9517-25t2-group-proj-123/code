import cv2 as cv
import numpy as np
import torch
from scipy import ndimage as ndi


def initial_segmentation_refinement(seg, seg_thresh=0.5, min_area=20):
    # seg: torch.Tensor, shape (1, 1, H, W)
    seg_np = np.squeeze(seg.detach().cpu().numpy())  # (H, W)
    bin_seg = (seg_np > seg_thresh).astype(np.uint8)
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(bin_seg, connectivity=8)
    mask = np.zeros_like(seg_np, dtype=np.uint8)
    for i in range(1, n_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 1

    return torch.from_numpy(mask).to(seg.device).view_as(seg)

def apply_postprocess(prediction, process):
    if process == 'initial_segmentation_refinement':
        return initial_segmentation_refinement(
            prediction,
            seg_thresh=0.5,
            min_area=50
        )
    else:
        raise ValueError(f"Unknown postprocess method: {process}")