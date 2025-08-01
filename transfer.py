import os
import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from tqdm import tqdm
import numpy as np
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

from config.config import get_config
from data.loader import prepare_dataloaders
from model.loss import BCEDiceLoss

# Set cache directories before importing models
os.environ['TORCH_HOME'] = '/srv/scratch/CRUISE/shawn/cache/torch'
os.environ['TORCH_HUB_DIR'] = '/srv/scratch/CRUISE/shawn/cache/torch/hub'
os.makedirs('/srv/scratch/CRUISE/shawn/cache/torch/hub', exist_ok=True)


def get_torchvision_model():
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
    
    model = segmentation.deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    )
    
    # Replace classifier for binary segmentation
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    
    # Freeze backbone and most of classifier
    for name, param in model.named_parameters():
        # Only train the final classification layer
        if name.startswith('classifier.4'):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
    
    return model


def train(model, train_loader, optimizer, loss_fn, epochs, device):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Take only RGB channels
            if inputs.shape[1] > 3:
                inputs = inputs[:, :3, :, :]
            
            # Forward pass
            outputs = model(inputs)['out']  # DeepLabV3 returns dict with 'out' key
            
            # Resize to match labels if needed
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = nn.functional.interpolate(
                    outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )
            
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}")


def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    total_iou = total_precision = total_recall = total_f1 = count = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if inputs.shape[1] > 3:
                inputs = inputs[:, :3, :, :]
            
            outputs = model(inputs)['out']
            
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = nn.functional.interpolate(
                    outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )
            
            pred = torch.sigmoid(outputs) > 0.5
            gt_mask = labels > 0.5

            pred_flat = pred.cpu().numpy().astype(np.uint8).flatten()
            gt_flat = gt_mask.cpu().numpy().astype(np.uint8).flatten()

            if gt_flat.sum() == 0 and pred_flat.sum() == 0:
                continue

            total_iou += jaccard_score(gt_flat, pred_flat, zero_division=0)
            total_precision += precision_score(gt_flat, pred_flat, zero_division=0)
            total_recall += recall_score(gt_flat, pred_flat, zero_division=0)
            total_f1 += f1_score(gt_flat, pred_flat, zero_division=0)
            count += 1

    results = {
        'mean_iou': total_iou / count if count > 0 else 0,
        'precision': total_precision / count if count > 0 else 0,
        'recall': total_recall / count if count > 0 else 0,
        'f1': total_f1 / count if count > 0 else 0,
    }
    print(f"Eval: | mIoU | F1 | Precision | Recall |")
    print(f"      | {results['mean_iou']:.3f} | {results['f1']:.3f} | {results['precision']:.3f} | {results['recall']:.3f} |")
    return results


def main():
    cfg = get_config()
    cfg['dataset']['nrg'] = False
    print(f'Cfg: {cfg}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_torchvision_model()
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['trainer']['learning_rate']
    )
    loss_fn = BCEDiceLoss()
    train_loader, test_loader = prepare_dataloaders(cfg)

    epochs = cfg['trainer']['epochs']
    
    print(f"Starting training for {epochs} epochs...")
    train(model, train_loader, optimizer, loss_fn, epochs, device)
    
    print("Evaluating trained model...")
    results = evaluate(model, test_loader, loss_fn, device)
    
    model_path = cfg['output']['model_save_path'].replace('.pth', '_deeplabv3.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()