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

# not used
def tree_iou(pred_inst, gt_inst):
    pred_lbl = np.unique(pred_inst)[1:]
    gt_lbl = np.unique(gt_inst)[1:]
    
    if len(gt_lbl) == 0:
        return 1.0 if len(pred_lbl) == 0 else 0.0
    
    total_iou = 0.0
    matched = set()
    
    for p_lbl in pred_lbl:
        p_mask = (pred_inst == p_lbl)
        best_iou = 0.0
        best_gt = None
        
        for g_lbl in gt_lbl:
            if g_lbl in matched:
                continue
                
            g_mask = (gt_inst == g_lbl)
            
            inter = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            
            if union > 0:
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g_lbl
        
        if best_gt is not None:
            total_iou += best_iou
            matched.add(best_gt)
    
    return total_iou / len(gt_lbl)

# not used
def centroid_localization_error(pred_inst, gt_inst):
    def extract_centroids(inst):
        lbls = np.unique(inst)[1:]
        cents = []
        
        for lbl in lbls:
            mask = (inst == lbl)
            if mask.sum() > 0:
                center = ndi.center_of_mass(mask)
                if len(center) == 2:
                    cents.append([center[1], center[0]])
        
        return np.array(cents) if cents else np.empty((0, 2))
    
    pred_cents = extract_centroids(pred_inst)
    gt_cents = extract_centroids(gt_inst)
    
    if len(gt_cents) == 0:
        return 0.0 if len(pred_cents) == 0 else float('inf')
    
    if len(pred_cents) == 0:
        return float('inf')
    
    if len(pred_cents) > 0 and len(gt_cents) > 0:
        dists = cdist(pred_cents, gt_cents)
        
        matched_dists = []
        used = set()
        
        for i in range(len(pred_cents)):
            min_dist = float('inf')
            best_j = None
            
            for j in range(len(gt_cents)):
                if j not in used and dists[i, j] < min_dist:
                    min_dist = dists[i, j]
                    best_j = j
            
            if best_j is not None:
                matched_dists.append(min_dist)
                used.add(best_j)
        
        if matched_dists:
            rmse = np.sqrt(np.mean(np.array(matched_dists) ** 2))
        else:
            rmse = float('inf')
    else:
        rmse = float('inf')
    
    return rmse