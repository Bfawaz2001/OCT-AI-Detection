import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class RetinalDiseaseModel(nn.Module):
    """
    A Convolutional Neural Network (CNN) for multi-label classification
    of retinal diseases using a modified ResNet-18 architecture.
    """

    def __init__(self, num_classes=28):
        """
        Initializes the model with a pre-trained ResNet-18 backbone and a custom output layer.

        Args:
            num_classes (int): Number of output classes (disease labels).
        """
        super(RetinalDiseaseModel, self).__init__()

        # Load ResNet-18 model with updated weights parameter
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify the final fully connected layer to match the number of classes
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes),
            nn.Sigmoid()  # Sigmoid activation for multi-label classification
        )

    def forward(self, x):
        """
        Defines the forward pass through the network.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor with probabilities for each class.
        """
        return self.model(x)


# Example usage
if __name__ == "__main__":
    model = RetinalDiseaseModel(num_classes=28)
    sample_input = torch.randn(8, 3, 224, 224)  # Batch of 8 images, 3 channels, 224x224
    output = model(sample_input)
    print("Output shape:", output.shape)  # Expected: [8, 28]
