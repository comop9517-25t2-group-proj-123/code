"""
Trainer module for modular training pipeline
"""
import torch
from tqdm import tqdm

from model.loss import compute_total_loss


class Trainer:
    """Handles training and validation loops"""
    
    def __init__(self, model, optimizer, device, lambda_dice=1.0, lambda_centroid=1.0, 
                 lambda_hybrid=1.0, lambda_boundary=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_dice = lambda_dice
        self.lambda_centroid = lambda_centroid
        self.lambda_hybrid = lambda_hybrid
        self.lambda_boundary = lambda_boundary
        
        # Training history
        self.train_history = []
        self.val_history = []
    
    def run_epoch(self, data_loader, is_training=True):
        """Run one epoch of training or validation"""
        if is_training:
            self.model.train()
            phase_name = "Train"
        else:
            self.model.eval()
            phase_name = "Val"
        
        total_loss = 0
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc=f"{phase_name}")):
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                # print(images.shape, labels.shape, outputs.shape)
                # print(images.dtype, labels.dtype, outputs.dtype)
                # Compute loss
                loss = compute_total_loss(
                    outputs, labels, 
                    lambda_dice=self.lambda_dice,
                    lambda_centroid=self.lambda_centroid,
                    lambda_hybrid=self.lambda_hybrid,
                    lambda_boundary=self.lambda_boundary
                )
                
                if is_training:
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item()
        
        # Average loss
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def train_epoch(self, train_loader):
        """Run training epoch"""
        return self.run_epoch(train_loader, is_training=True)
    
    def validate_epoch(self, val_loader):
        """Run validation epoch"""
        return self.run_epoch(val_loader, is_training=False)
    
    def fit(self, train_loader, val_loader, epochs):
        """Full training loop"""
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            self.train_history.append({'total_loss': train_loss})
            
            # Validation phase
            val_loss = self.validate_epoch(val_loader)
            self.val_history.append({'total_loss': val_loss})
            
            # Print results
            self.print_epoch_results(epoch+1, train_loss, val_loss)
    
    def print_epoch_results(self, epoch, train_loss, val_loss):
        """Print formatted epoch results"""
        print(f"[Epoch {epoch}] Training Loss: {train_loss:.4f}")
        print(f"[Epoch {epoch}] Validation Loss: {val_loss:.4f}")
    
    def save_model(self, path):
        """Save model state dict"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved as '{path}'")
    
    def get_history(self):
        """Get training history"""
        return {
            'train': self.train_history,
            'val': self.val_history
        }
