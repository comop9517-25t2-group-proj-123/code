import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Ensure pred is sigmoid-activated for BCE with logits
        pred = torch.sigmoid(pred) if pred.max() > 1 else pred

        # Flatten batch-wise: (B, C, H, W) → (B, -1)
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()


def compute_segmentation_loss(pred, seg_gt, lambda_dice=1.0):
    """
    Compute segmentation loss combining BCE and Dice loss
    
    Args:
        pred: (B, 1, H, W) - model prediction (logits)
        seg_gt: (B, 1, H, W) - ground truth segmentation mask
        lambda_dice: weight for dice loss
    
    Returns:
        seg_loss: combined segmentation loss
    """
    bce_loss = nn.BCEWithLogitsLoss()(pred, seg_gt)
    dice_loss = DiceLoss()(pred, seg_gt)
    
    seg_loss = bce_loss + lambda_dice * dice_loss
    
    return seg_loss


def compute_centroid_loss(pred, centroid_gt):
    """
    Compute centroid localization loss
    
    Args:
        pred: (B, 1, H, W) - model prediction (logits)
        centroid_gt: (B, 1, H, W) - ground truth centroid map
    
    Returns:
        centroid_loss: MSE loss between sigmoid(pred) and centroid_gt
    """
    mse_loss = nn.MSELoss()
    pred_sigmoid = torch.sigmoid(pred)
    centroid_loss = mse_loss(pred_sigmoid, centroid_gt)
    
    return centroid_loss


def compute_hybrid_sdt_loss(pred, hybrid_gt, lambda_boundary=1.0):
    """
    Compute hybrid SDT-boundary loss with masking
    
    Args:
        pred: (B, 1, H, W) - model prediction (logits)
        hybrid_gt: (B, 1, H, W) - ground truth hybrid SDT map
        lambda_boundary: weight for boundary loss
    
    Returns:
        hybrid_loss: combined SDT and boundary loss
    """
    smooth_l1_loss = nn.SmoothL1Loss()
    l1_loss = nn.L1Loss()
    
    # Create masks for different regions
    sdt_mask = hybrid_gt > -1      # Inside and outside regions (SDT)
    boundary_mask = hybrid_gt == -1  # Boundary regions
    
    device = pred.device
    pred_tanh = torch.tanh(pred)
    
    # SDT loss for non-boundary regions
    if sdt_mask.sum() > 0:
        sdt_loss = smooth_l1_loss(pred_tanh[sdt_mask], hybrid_gt[sdt_mask])
    else:
        sdt_loss = torch.tensor(0.0, device=device)
    
    # Boundary loss for boundary regions
    if boundary_mask.sum() > 0:
        boundary_loss = l1_loss(pred_tanh[boundary_mask], hybrid_gt[boundary_mask])
    else:
        boundary_loss = torch.tensor(0.0, device=device)
    
    # Combined hybrid loss
    hybrid_loss = sdt_loss + lambda_boundary * boundary_loss
    
    return hybrid_loss


def compute_total_loss(pred, labels, lambda_dice=1.0, lambda_centroid=1.0, 
                      lambda_hybrid=1.0, lambda_boundary=1.0):

    seg_loss = compute_segmentation_loss(pred[:, 0], labels[:, 0], lambda_dice)
    centroid_loss = compute_centroid_loss(pred[:, 1], labels[:, 1])
    hybrid_loss = compute_hybrid_sdt_loss(pred[:, 2], labels[:, 2], lambda_boundary)
    
    # Combine losses
    total_loss = seg_loss + lambda_centroid * centroid_loss + lambda_hybrid * hybrid_loss
    
    return total_loss