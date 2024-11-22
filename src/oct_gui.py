import os
import torch
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
from torchvision import transforms
from model import RetinalDiseaseModel

# Configure paths
MODEL_PATH = "../outputs/models/retinal_disease_model_v2.pth"
DEVICE = torch.device("cpu")  # Ensure we use CPU

# Load model
def load_model():
    model = RetinalDiseaseModel(num_diseases=28).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

MODEL = load_model()

# Transformations for the input image
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the GUI class
class OCTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCT Disease Detection")
        self.root.geometry("600x400")
        self.image_path = None
        self.image_label = Label(root, text="No image uploaded.", width=40, height=10)
        self.image_label.pack(pady=10)

        # Buttons
        self.upload_button = Button(root, text="Upload Image", command=self.upload_image, width=20)
        self.upload_button.pack(pady=5)

        self.analyze_button = Button(root, text="Analyze Image", command=self.analyze_image, width=20)
        self.analyze_button.pack(pady=5)

        self.result_label = Label(root, text="Results will be displayed here.", wraplength=500, justify="left")
        self.result_label.pack(pady=10)

    def upload_image(self):
        """
        Allows the user to upload an image file and displays it in the GUI.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        self.image_path = file_path

        # Display the image
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        self.display_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.display_image, text="")
        self.image_label.image = self.display_image

        # Reset results
        self.result_label.configure(text="Results will be displayed here.")

    def analyze_image(self):
        """
        Analyzes the uploaded image using the trained model and displays results.
        """
        if not self.image_path:
            messagebox.showerror("Error", "No image uploaded.")
            return

        try:
            # Load and preprocess the image
            image = Image.open(self.image_path).convert("RGB")
            input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

            # Predict using the model
            with torch.no_grad():
                disease_risk, disease_outputs = MODEL(input_tensor)

            # Process results
            disease_risk = torch.sigmoid(disease_risk).item()
            disease_outputs = torch.sigmoid(disease_outputs).squeeze().tolist()

            # Interpret the results
            results = self.interpret_results(disease_risk, disease_outputs)
            self.result_label.configure(text=results)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def interpret_results(self, disease_risk, disease_outputs):
        """
        Interprets the model's predictions and formats them for display.

        Args:
            disease_risk (float): Probability of a disease being present.
            disease_outputs (list): Probabilities for each specific disease.

        Returns:
            str: Formatted results.
        """
        diseases = [
            "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
            "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
            "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"
        ]

        threshold = 0.5  # Default threshold for disease presence
        result_text = f"Disease Risk: {disease_risk:.2f}\n"

        if disease_risk > threshold:
            result_text += "Potential Disease Detected:\n"
            for i, prob in enumerate(disease_outputs):
                if prob > threshold:
                    result_text += f"  - {diseases[i]}: {prob:.2f}\n"
        else:
            result_text += "No disease detected."

        return result_text

# Run the application
if __name__ == "__main__":
    root = Tk()
    app = OCTApp(root)
    root.mainloop()
