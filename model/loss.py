import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    """
    Example usage:
        criterion = BCEDiceLoss()
        loss = criterion(pred_mask, gt_mask)
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        # BCE Loss (expects logits)
        bce_loss = self.bce(pred, target)
        
        # Dice Loss (expects probabilities, so use sigmoid)
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred_prob * target).sum()
        dice = (2. * intersection + smooth) / (pred_prob.sum() + target.sum() + smooth)
        dice_loss = 1 - dice
        
        # Weighted sum
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

