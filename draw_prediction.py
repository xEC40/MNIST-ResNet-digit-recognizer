import torch
import numpy as np
import pygame
from PIL import Image
from torchvision import transforms
from nn import ResNet
from ui import UI

class DigitRecognizer:
    def __init__(self, model_path='best_model.pth'):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = ResNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Define transformations (same as in training)
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        print("Model loaded successfully")
    
    def preprocess_image(self, pixels):
        """
        Preprocess the drawn image to match MNIST format.
        
        Args:
            pixels: numpy array of pixel values (height, width)
            
        Returns:
            torch.Tensor: Preprocessed image tensor ready for the model
        """
        # Find the bounding box of the digit
        rows = np.any(pixels, axis=1)
        cols = np.any(pixels, axis=0)
        
        # If the image is empty (no drawing), return a blank image
        if not np.any(rows) or not np.any(cols):
            blank = np.zeros((28, 28), dtype=np.uint8)
            img = Image.fromarray(blank)
            img_tensor = self.transform(img).unsqueeze(0)
            return img_tensor.to(self.device)
        
        # Get the non-empty row and column indices
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding to ensure we don't cut off any part of the digit
        padding = 2
        rmin = max(0, rmin - padding)
        rmax = min(pixels.shape[0] - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(pixels.shape[1] - 1, cmax + padding)
        
        # Extract the digit
        digit = pixels[rmin:rmax+1, cmin:cmax+1]
        
        # Create a square canvas with padding
        size = max(digit.shape[0], digit.shape[1])
        square_digit = np.zeros((size, size), dtype=np.uint8)
        
        # Center the digit in the square canvas
        y_offset = (size - digit.shape[0]) // 2
        x_offset = (size - digit.shape[1]) // 2
        square_digit[y_offset:y_offset+digit.shape[0], x_offset:x_offset+digit.shape[1]] = digit
        
        # Convert to PIL Image
        img = Image.fromarray(square_digit)
        
        # Apply transformations
        img_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
        
        return img_tensor.to(self.device)
    
    def predict(self, pixels):
        """
        Make a prediction on the drawn image.
        
        Args:
            pixels: numpy array of pixel values (height, width)
            
        Returns:
            tuple: (predicted_digit, probabilities)
        """
        # Preprocess image
        img_tensor = self.preprocess_image(pixels)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.exp(output).squeeze().cpu().numpy()
            
            # Normalize to sum to 1
            probabilities = probabilities / np.sum(probabilities)
            
            # Get predicted digit
            predicted_digit = np.argmax(probabilities)
        
        return predicted_digit, probabilities

def main():
    # Initialize the digit recognizer
    recognizer = DigitRecognizer()
    
    # Define prediction callback for UI
    def prediction_callback(pixels):
        return recognizer.predict(pixels)
    
    # Initialize and run UI
    ui = UI(width=800, height=600)
    ui.run(prediction_callback)

if __name__ == "__main__":
    main()
