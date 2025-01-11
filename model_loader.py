import torch
from torchvision import transforms
from model_trainer import ModelTrainer  # Import your model class
from config import MODEL_PATH, MODEL_NAME, NUM_CLASSES,PREDICTION_THRESHOLD


class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.model = ModelTrainer(MODEL_NAME, NUM_CLASSES)
        self.load_model()

    def load_model(self):
        try:
            # Load the checkpoint with correct device mapping
            checkpoint = torch.load(MODEL_PATH,
                                    map_location=self.device)
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.model.to(self.device)
            self.model.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, frame):
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model.model(input_tensor)
            prediction = output > PREDICTION_THRESHOLD  # Apply thresholding
        print(f"Model raw output: {output}")
        print(f"Thresholded predictions: {prediction}")
        return prediction