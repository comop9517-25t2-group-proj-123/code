"""
Clean modularized training script using Trainer class with configuration dictionary
"""
import torch

from data.loader import prepare_dataloaders
from engine.trainer import Trainer
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with config
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg['trainer']['learning_rate']
    )

    # Prepare dataloaders with config
    train_loader, val_loader, test_loader = prepare_dataloaders(
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

    # Initialize trainer with config
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
    )

    # Train the model
    epochs = cfg['trainer']['epochs']
    print(f"\nStarting training for {epochs} epochs...")
    trainer.fit(train_loader, val_loader, epochs)

    # Save the trained model
    model_path = cfg['output']['model_save_path']
    trainer.save_model(model_path)
    
    # Get training history
    history = trainer.get_history()
    print(f"\nTraining completed!")
    print(f"Final train loss: {history['train'][-1]['total_loss']:.4f}")
    print(f"Final val loss: {history['val'][-1]['total_loss']:.4f}")


if __name__ == "__main__":
    main()