"""
Test script with configurable postprocessing methods
"""
import torch

from data.loader import prepare_dataloaders
from engine.tester import Tester
from model.unet import UNet
from config.config import get_config


def main():
    # Load configuration
    cfg = get_config()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with config
    model = UNet(
        in_channels=cfg['model']['in_channels'],
        n_classes=cfg['model']['n_classes'],
        depth=cfg['model']['depth'],
        padding=True,
        batch_norm=True,
    ).to(device)

    # Load trained model
    model_path = cfg['output']['model_save_path']
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")

    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare test dataloader
    _, _, test_loader = prepare_dataloaders(
        data_root=cfg['dataset']['data_root'],
        val_ratio=cfg['dataset']['val_ratio'],
        test_ratio=cfg['dataset']['test_ratio'],
        patch_size=cfg['dataset']['patch_size'],
        stride=cfg['dataset']['stride'],
        train_batch_size=cfg['dataloader']['train_batch_size'],
        val_batch_size=cfg['dataloader']['val_batch_size'],
        test_batch_size=cfg['dataloader']['test_batch_size'],
        num_workers=cfg['dataloader']['num_workers']
    )

    # Create tester with config
    tester = Tester(model, device, cfg)
    
    print(f"Using postprocessing methods: {cfg['test']['postprocess_methods']}")
    
    # Evaluate model
    results = tester.evaluate(test_loader)
    

if __name__ == "__main__":
    main()