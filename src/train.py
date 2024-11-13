import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_preprocessing import create_dataloaders
from model import RetinalDiseaseModel

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Paths to the data
train_csv = "../data/train/RFMiD_Training_Labels.csv"
val_csv = "../data/val/RFMiD_Validation_Labels.csv"
test_csv = "../data/test/RFMiD_Testing_Labels.csv"
train_dir = "../data/train/images"
val_dir = "../data/val/images"
test_dir = "../data/test/images"

# Load Data
train_loader, val_loader, test_loader = create_dataloaders(
    train_csv, val_csv, test_csv, train_dir, val_dir, test_dir, batch_size=batch_size
)

# Initialize the model, loss function, and optimizer
model = RetinalDiseaseModel(num_classes=28).to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_one_epoch(loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def validate(loader, model, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(loader)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch(train_loader, model, criterion, optimizer)
    val_loss = validate(val_loader, model, criterion)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "../outputs/models/retinal_disease_model.pth")
print("Model saved at '../outputs/models/retinal_disease_model.pth'")
