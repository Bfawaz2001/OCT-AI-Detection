import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from data_preprocessing import create_dataloaders
from model import RetinalDiseaseModel
from datetime import datetime

# Set device
device = torch.device("cpu")  # Using CPU for training

# Paths to data files
train_csv = "../data/train/RFMiD_Training_Labels.csv"
val_csv = "../data/val/RFMiD_Validation_Labels.csv"
train_dir = "../data/train/images"
val_dir = "../data/val/images"

# Data transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def compute_class_weights(labels_csv, num_classes):
    import pandas as pd
    df = pd.read_csv(labels_csv)
    class_counts = df.iloc[:, 2:].sum(axis=0)
    total_samples = len(df)
    class_weights = total_samples / (num_classes * class_counts)
    return torch.tensor(class_weights.values, dtype=torch.float32).to(device)

class CombinedLoss(nn.Module):
    def __init__(self, disease_weight=None, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.binary_loss = nn.BCEWithLogitsLoss(pos_weight=disease_weight)
        self.multilabel_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def forward(self, outputs, targets):
        disease_risk_out, disease_labels_out = outputs
        disease_risk_targets, disease_labels_targets = targets
        loss_risk = self.binary_loss(disease_risk_out, disease_risk_targets.unsqueeze(1))
        loss_labels = self.multilabel_loss(disease_labels_out, disease_labels_targets)
        return 0.3 * loss_risk + 0.7 * loss_labels

def train_one_epoch(loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, disease_risks, labels in loader:
        images, disease_risks, labels = images.to(device), disease_risks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, (disease_risks, labels))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(loader, model, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, disease_risks, labels in loader:
            images, disease_risks, labels = images.to(device), disease_risks.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, (disease_risks, labels))
            running_loss += loss.item()
    return running_loss / len(loader)

def main():
    num_classes = 28
    learning_rate = 0.001  # Match Version 2's learning rate
    batch_size = 20  # Match Version 2's batch size
    num_epochs = 40
    patience = 5

    disease_class_weights = compute_class_weights(train_csv, num_classes)
    disease_risk_weight = torch.tensor([1.0], dtype=torch.float32).to(device)

    train_loader, val_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        train_dir=train_dir,
        val_dir=val_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=batch_size
    )

    model = RetinalDiseaseModel(num_diseases=num_classes).to(device)
    criterion = CombinedLoss(disease_weight=disease_risk_weight, class_weights=disease_class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR by half every 10 epochs

    best_val_loss = float('inf')
    early_stop_counter = 0
    checkpoint_dir = "../outputs/models/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}] - {datetime.now()}")
        print(f"Current Learning Rate: {current_lr:.6f}")

        train_loss = train_one_epoch(train_loader, model, criterion, optimizer)
        val_loss = validate(val_loader, model, criterion)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "retinal_disease_model_v9.pth"))
            print("Model saved with improved validation loss.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
