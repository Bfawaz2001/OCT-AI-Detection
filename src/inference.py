import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import RetinalDiseaseModel


# Load model
def load_model(model_path, num_classes=28):
    model = RetinalDiseaseModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function for making predictions on a single image
def predict_image(model, image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)
        predictions = output.squeeze().numpy()

    # Interpret results: apply a threshold (e.g., 0.5) to get binary predictions
    disease_risk = predictions > 0.5
    return disease_risk, predictions  # Return both binary and probability scores


# Example usage
if __name__ == "__main__":
    # Path to the saved model and new image
    model_path = "../outputs/models/retinal_disease_model_v4.pth"
    image_path = "../data/test/images/2.png"  # Replace with your image path

    # Load the trained model
    model = load_model(model_path)

    # Predict disease presence
    disease_risk, probabilities = predict_image(model, image_path)

    print("Disease Presence (1=True, 0=False):", disease_risk)
    print("Probabilities for each disease:", probabilities)
