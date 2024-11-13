import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Define transformations for image resizing and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize RGB channels
])


class RetinalDataset(Dataset):
    """
    Custom Dataset class for loading retinal OCT images and their multi-label annotations.

    Attributes:
        labels_df (pd.DataFrame): DataFrame containing image IDs and multi-label targets.
        image_dir (str): Directory with all the images.
        transform (callable, optional): Optional transformations to be applied on a sample.
    """

    def __init__(self, labels_df, image_dir, transform=None):
        """
        Initializes the RetinalDataset with labels DataFrame and image directory.

        Args:
            labels_df (pd.DataFrame): DataFrame containing image IDs and multi-label targets.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transformations to be applied on a sample.
        """
        self.labels_df = labels_df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        Retrieves a sample image and its corresponding labels.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, labels) where image is the processed image tensor,
                   and labels is a tensor of binary multi-labels for the sample.
        """
        # Get image ID and construct file path
        img_id = self.labels_df.iloc[idx, 0]  # Assuming the ID column is the first column
        img_path = os.path.join(self.image_dir, f"{img_id}.png")

        # Load and convert the image to RGB
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if defined
        if self.transform:
            image = self.transform(image)

        # Extract labels from DataFrame, skipping ID and Disease_Risk columns
        labels = torch.tensor(self.labels_df.iloc[idx, 2:].values, dtype=torch.float32)

        return image, labels


def create_dataloaders(train_csv, val_csv, test_csv, train_dir, val_dir, test_dir, batch_size=32):
    """
    Creates DataLoaders for the training, validation, and test datasets.

    Args:
        train_csv (str): Path to the training labels CSV file.
        val_csv (str): Path to the validation labels CSV file.
        test_csv (str): Path to the test labels CSV file.
        train_dir (str): Path to the directory containing training images.
        val_dir (str): Path to the directory containing validation images.
        test_dir (str): Path to the directory containing test images.
        batch_size (int, optional): Batch size for DataLoader. Default is 32.

    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoaders for each dataset.
    """
    # Load label DataFrames
    train_labels = pd.read_csv(train_csv)
    val_labels = pd.read_csv(val_csv)
    test_labels = pd.read_csv(test_csv)

    # Initialize datasets
    train_dataset = RetinalDataset(train_labels, train_dir, transform=transform)
    val_dataset = RetinalDataset(val_labels, val_dir, transform=transform)
    test_dataset = RetinalDataset(test_labels, test_dir, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(
        "../data/train/RFMiD_Training_Labels.csv",
        "../data/val/RFMiD_Validation_Labels.csv",
        "../data/test/RFMiD_Testing_Labels.csv",
        "../data/train/images",
        "../data/val/images",
        "../data/test/images"
    )

    # Check one batch
    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

