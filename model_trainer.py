import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import os


class ModelTrainer:
    def __init__(self, model_name, num_classes, device='cpu'):
        self.device = device
        self.num_classes = num_classes

        # Dictionary of available models
        self.available_models = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'mobilenet_v2': models.mobilenet_v2,
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1
        }

        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.available_models.keys())}")

        # Initialize the selected model
        self.model = self.available_models[model_name](pretrained=True)

        # Modify final layer for multi-label classification
        if 'resnet' in model_name:
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, num_classes),
                nn.Sigmoid()
            )
        elif 'mobilenet' in model_name:
            self.model.classifier[1] = nn.Sequential(
                nn.Linear(self.model.classifier[1].in_features, num_classes),
                nn.Sigmoid()
            )
        elif 'efficientnet' in model_name:
            self.model.classifier[1] = nn.Sequential(
                nn.Linear(self.model.classifier[1].in_features, num_classes),
                nn.Sigmoid()
            )

        self.model.to(device)
        self.model_name = model_name

    def train_model(self, train_loader, val_loader, lr=0.0001, epochs=5, save_dir='model_checkpoints'):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        history = defaultdict(list)
        best_val_loss = float('inf')

        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, f'{self.model_name}_best.pth')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            start_time = time.time()

            train_loss = train_epoch(self.model, train_loader, criterion, optimizer, self.device)
            val_loss = validate(self.model, val_loader, criterion, self.device)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            epoch_time = time.time() - start_time
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Time: {epoch_time:.2f}s')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f'Validation loss improved to {val_loss:.4f}. Saving model...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, best_model_path)

        return history, best_val_loss, best_model_path

    def plot_training_results(self, history):
        plt.figure(figsize=(12, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'{self.model_name} Training Results')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (BCE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Keep your existing train_epoch and validate functions
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(test_loader)