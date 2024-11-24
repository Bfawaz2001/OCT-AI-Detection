import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import RetinalDataset, create_dataloaders
from model import RetinalDiseaseModel
from torchvision import transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
model_path = "../outputs/models/retinal_disease_model_v15.pth"
test_csv = "../data/test/RFMiD_Testing_Labels.csv"
test_dir = "../data/test/images"

# Transformation for test set
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def evaluate(loader, model, risk_threshold=0.5, disease_threshold=0.2):
    """
    Evaluate the model's performance on a test dataset.

    Args:
        loader: DataLoader for the test set.
        model: Trained model for evaluation.
        risk_threshold: Threshold for binary classification (disease risk).
        disease_threshold: Threshold for multi-label classification (specific diseases).

    Returns:
        Metrics for disease risk and specific disease classifications.
    """
    model.eval()
    all_disease_risks = []
    all_pred_disease_risks = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, disease_risks, labels in loader:
            images, disease_risks, labels = images.to(device), disease_risks.to(device), labels.to(device)
            risk_outputs, disease_outputs = model(images)

            # Disease Risk Prediction
            pred_disease_risks = (risk_outputs > risk_threshold).float()
            all_disease_risks.extend(disease_risks.cpu().numpy())
            all_pred_disease_risks.extend(pred_disease_risks.cpu().numpy())

            # Specific Disease Prediction
            pred_disease_labels = (disease_outputs > disease_threshold).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_disease_labels.cpu().numpy())

    # Disease Risk Metrics
    disease_risk_accuracy = accuracy_score(all_disease_risks, all_pred_disease_risks)
    disease_risk_precision = precision_score(all_disease_risks, all_pred_disease_risks)
    disease_risk_recall = recall_score(all_disease_risks, all_pred_disease_risks)
    disease_risk_f1 = f1_score(all_disease_risks, all_pred_disease_risks)

    # Specific Disease Metrics
    specific_accuracy = accuracy_score(all_labels, all_preds)
    specific_precision = precision_score(all_labels, all_preds, average="macro", zero_division=1)
    specific_recall = recall_score(all_labels, all_preds, average="macro", zero_division=1)
    specific_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=1)

    return (disease_risk_accuracy, disease_risk_precision, disease_risk_recall, disease_risk_f1,
            specific_accuracy, specific_precision, specific_recall, specific_f1)


def main():
    # Load model
    model = RetinalDiseaseModel(num_diseases=28).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Create test DataLoader
    test_labels = pd.read_csv(test_csv)
    test_dataset = RetinalDataset(test_labels, test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # Evaluate with different thresholds
    for risk_threshold in [0.5, 0.6, 0.7]:
        for disease_threshold in [0.0001, 0.1,0.2, 0.6]:
            print(f"\nEvaluating with Risk Threshold: {risk_threshold}, Disease Threshold: {disease_threshold}")
            (disease_risk_acc, disease_risk_prec, disease_risk_recall, disease_risk_f1,
             specific_acc, specific_prec, specific_recall, specific_f1) = evaluate(
                test_loader, model, risk_threshold=risk_threshold, disease_threshold=disease_threshold
            )

            print("Disease Risk Metrics:")
            print(f"  Accuracy: {disease_risk_acc:.4f}")
            print(f"  Precision: {disease_risk_prec:.4f}")
            print(f"  Recall: {disease_risk_recall:.4f}")
            print(f"  F1 Score: {disease_risk_f1:.4f}")
            print("\nSpecific Disease Metrics:")
            print(f"  Accuracy: {specific_acc:.4f}")
            print(f"  Precision: {specific_prec:.4f}")
            print(f"  Recall: {specific_recall:.4f}")
            print(f"  F1 Score: {specific_f1:.4f}")


if __name__ == "__main__":
    main()
