import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_coeff

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss
    
    Example usage:
        criterion = BCEDiceLoss()
        loss = criterion(pred_mask, gt_mask)
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice

"""
Adapted from "Dual-Task Learning for Dead Tree Detection and Segmentation with Hybrid  Self-Attention U-Nets in Aerial Imagery"
Code at https://github.com/Global-Ecosystem-Health-Observatory/TreeMort
"""
class HybridLoss(nn.Module):
    """
    Combined loss function for multi-task learning with segmentation, centroid detection, and SDT prediction.
    
    Example usage:
        criterion = HybridLoss(lambda_dice=1.0, lambda_centroid=1.0, lambda_hybrid=1.0, lambda_boundary=1.0)
        loss = criterion(pred, labels)
    """
    def __init__(self, lambda_dice=1.0, lambda_centroid=1.0, lambda_hybrid=1.0, lambda_boundary=1.0):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_centroid = lambda_centroid
        self.lambda_hybrid = lambda_hybrid
        self.lambda_boundary = lambda_boundary
        
        # Initialize loss components
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, labels):
        """
        Args:
            pred: (B, 3, H, W) - model predictions [segmentation, centroid, hybrid_sdt]
            labels: (B, 3, H, W) - ground truth labels [segmentation, centroid, hybrid_sdt]
        
        Returns:
            total_loss: combined loss value
        """
        seg_loss = self._segmentation_loss(pred[:, 0], labels[:, 0])
        
        centroid_loss = self._centroid_loss(pred[:, 1], labels[:, 1])
        
        hybrid_loss = self._sdt_loss(pred[:, 2], labels[:, 2])
        
        total_loss = (seg_loss + 
                     self.lambda_centroid * centroid_loss + 
                     self.lambda_hybrid * hybrid_loss)
        
        return total_loss
    
    def _segmentation_loss(self, pred, seg_gt):
        """Compute segmentation loss combining BCE and Dice loss"""
        bce_loss = self.bce_loss(pred, seg_gt)
        dice_loss = self.dice_loss(pred, seg_gt)
        return bce_loss + self.lambda_dice * dice_loss
    
    def _centroid_loss(self, pred, centroid_gt):
        """Compute centroid localization loss"""
        pred_sigmoid = torch.sigmoid(pred)
        return self.mse_loss(pred_sigmoid, centroid_gt)
    
    def _sdt_loss(self, pred, hybrid_gt):
        """Compute hybrid SDT-boundary loss with masking"""
        # Create masks for different regions
        sdt_mask = hybrid_gt > -1      # Inside and outside regions (SDT)
        boundary_mask = hybrid_gt == -1  # Boundary regions
        
        device = pred.device
        pred_tanh = torch.tanh(pred)
        
        # SDT loss for non-boundary regions
        if sdt_mask.sum() > 0:
            sdt_loss = self.smooth_l1_loss(pred_tanh[sdt_mask], hybrid_gt[sdt_mask])
        else:
            sdt_loss = torch.tensor(0.0, device=device)
        
        # Boundary loss for boundary regions
        if boundary_mask.sum() > 0:
            boundary_loss = self.l1_loss(pred_tanh[boundary_mask], hybrid_gt[boundary_mask])
        else:
            boundary_loss = torch.tensor(0.0, device=device)
        
        # Combined hybrid loss
        return sdt_loss + self.lambda_boundary * boundary_loss