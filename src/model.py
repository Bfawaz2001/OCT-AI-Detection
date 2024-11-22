import torch
import torch.nn as nn
import timm  # Library for pretrained models

class RetinalDiseaseModel(nn.Module):
    """
    A neural network for detecting disease presence and identifying specific retinal diseases
    from OCT images.

    Attributes:
        backbone (timm.models): Pretrained backbone model for feature extraction.
        disease_risk_head (nn.Sequential): Head for binary disease risk prediction.
        disease_classification_head (nn.Sequential): Head for multi-label disease classification.
    """

    def __init__(self, backbone='resnet50', num_diseases=28):
        """
        Initializes the RetinalDiseaseModel.

        Args:
            backbone (str): The name of the pretrained model to use as a backbone.
            num_diseases (int): The number of disease classes for multi-label classification.
        """
        super(RetinalDiseaseModel, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        num_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)

        # Head for predicting if a disease exists (binary classification)
        self.disease_risk_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # Head for identifying specific diseases (multi-label classification)
        self.disease_classification_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_diseases)
        )

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor representing batch of images.

        Returns:
            torch.Tensor: Output for disease risk prediction (binary classification).
            torch.Tensor: Output for specific disease classification (multi-label).
        """
        features = self.backbone(x)
        disease_risk_out = self.disease_risk_head(features)
        disease_labels_out = self.disease_classification_head(features)
        return disease_risk_out, disease_labels_out
