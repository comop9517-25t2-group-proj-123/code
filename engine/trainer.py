import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, optimizer, loss_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.train_history = []

    def fit(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred = self.model(images)
                loss = self.loss_fn(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            self.train_history.append({'epoch': epoch+1, 'loss': avg_loss})
            print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

    def save(self, save_path=None):
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved as '{save_path}'")

    def get_history(self):
        return {'train': self.train_history}