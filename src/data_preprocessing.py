import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2  # For loading images

# Transformation for data augmentation and normalization
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RetinalDataset(Dataset):
    """
    Custom dataset class for loading retinal OCT images and their multi-label annotations.

    Attributes:
        labels_df (pd.DataFrame): DataFrame containing the image IDs and labels.
        image_dir (str): Directory containing the images.
        transform (torchvision.transforms.Compose): Transformations to be applied to the images.
    """

    def __init__(self, labels_df, image_dir, transform=None):
        """
        Initializes the RetinalDataset.

        Args:
            labels_df (pd.DataFrame): DataFrame containing image IDs and labels.
            image_dir (str): Path to the directory containing images.
            transform (torchvision.transforms.Compose): Transformations to apply to the images.
        """
        self.labels_df = labels_df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Transformed image tensor.
            torch.Tensor: Binary disease risk label.
            torch.Tensor: Multi-label disease classification vector.
        """
        # Retrieve the image ID and construct its file path
        img_id = self.labels_df.iloc[idx]['ID']
        img_path = os.path.join(self.image_dir, f"{img_id}.png")

        # Load the image using OpenCV and convert to RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image {img_path} not found.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a PIL Image for compatibility with torchvision.transforms
        image = Image.fromarray(image)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Retrieve the disease risk and disease-specific labels
        disease_risk = torch.tensor(self.labels_df.iloc[idx]['Disease_Risk'], dtype=torch.float32)
        disease_labels = torch.tensor(self.labels_df.iloc[idx][2:].values, dtype=torch.float32)

        return image, disease_risk, disease_labels


def create_dataloaders(train_csv, val_csv, train_dir, val_dir, train_transform=None, val_transform=None, batch_size=32):
    """
    Creates DataLoaders for the training and validation datasets.

    Args:
        train_csv (str): Path to the training labels CSV file.
        val_csv (str): Path to the validation labels CSV file.
        train_dir (str): Path to the directory containing training images.
        val_dir (str): Path to the directory containing validation images.
        train_transform (torchvision.transforms.Compose): Transformations for the training dataset.
        val_transform (torchvision.transforms.Compose): Transformations for the validation dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: DataLoader for the training dataset.
        DataLoader: DataLoader for the validation dataset.
    """
    # Load the label CSV files
    train_labels = pd.read_csv(train_csv)
    val_labels = pd.read_csv(val_csv)

    # Define the datasets
    train_dataset = RetinalDataset(train_labels, train_dir, transform=train_transform)
    val_dataset = RetinalDataset(val_labels, val_dir, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Test the data loading process
if __name__ == "__main__":
    # Example file paths and directories
    train_csv = "../data/train/RFMiD_Training_Labels.csv"
    val_csv = "../data/val/RFMiD_Validation_Labels.csv"
    train_dir = "../data/train/images"
    val_dir = "../data/val/images"

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = default_transform

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        train_dir=train_dir,
        val_dir=val_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=32
    )

    # Check a batch of data
    images, disease_risks, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Disease Risk batch shape:", disease_risks.shape)
    print("Label batch shape:", labels.shape)
