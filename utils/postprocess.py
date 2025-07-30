import cv2 as cv
import numpy as np
from scipy import ndimage as ndi


def initial_segmentation_refinement(seg, seg_thresh=0.5, min_area=20):
    bin_seg = (seg > seg_thresh).astype(np.uint8)
    
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(bin_seg, connectivity=8)
    mask = np.zeros_like(bin_seg)

    for i in range(1, n_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 1
    
    return mask


def hybrid_filtering(mask, hybrid_map, hyb_thresh=0.5):
    boundary = (hybrid_map < hyb_thresh).astype(np.uint8)
    filtered = cv.bitwise_and(mask, boundary)
    return filtered


def centroid_marker_extraction(cent_map, sigma=2, filt_size=5, thresh=0.1):
    smooth = ndi.gaussian_filter(cent_map, sigma=sigma)
    local_max = (smooth == ndi.maximum_filter(smooth, size=filt_size)) & (smooth > thresh)
    markers, _ = ndi.label(local_max)
    return markers


def watershed_segmentation(mask, markers):
    dist = ndi.distance_transform_edt(mask)
    ws_input = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
    labels_ws = cv.watershed(ws_input, markers.astype(np.int32))
    instance = (labels_ws > 0).astype(np.uint8)
    return instance


