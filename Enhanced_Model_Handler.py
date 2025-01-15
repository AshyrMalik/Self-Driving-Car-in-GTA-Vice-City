import torch
from torchvision import transforms
import numpy as np
import cv2
from model_trainer import ModelTrainer
from config import MODEL_PATH, MODEL_NAME, NUM_CLASSES, PREDICTION_THRESHOLD


class LaneDetector:
    def __init__(self):
        self.previous_lanes = None

    def detect(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Define region of interest
        height, width = edges.shape
        mask = np.zeros_like(edges)
        vertices = np.array([[(0, height),
                              (width // 4, height // 2),
                              (3 * width // 4, height // 2),
                              (width, height)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        roi = cv2.bitwise_and(edges, mask)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(roi, 2, np.pi / 180, 20, minLineLength=20, maxLineGap=5)

        return self._process_lines(lines, frame.shape) if lines is not None else self.previous_lanes

    def _process_lines(self, lines, frame_shape):
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if abs(slope) < 0.5:  # Filter horizontal lines
                continue

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

        lane_info = self._average_lanes(left_lines, right_lines, frame_shape)
        self.previous_lanes = lane_info
        return lane_info

    def _average_lanes(self, left_lines, right_lines, frame_shape):
        height = frame_shape[0]

        if len(left_lines) > 0:
            left_avg = np.mean(left_lines, axis=0)
            left_x1 = int((height - left_avg[1]) / left_avg[0])
            left_x2 = int((height // 2 - left_avg[1]) / left_avg[0])
            left_lane = [left_x1, height, left_x2, height // 2]
        else:
            left_lane = None

        if len(right_lines) > 0:
            right_avg = np.mean(right_lines, axis=0)
            right_x1 = int((height - right_avg[1]) / right_avg[0])
            right_x2 = int((height // 2 - right_avg[1]) / right_avg[0])
            right_lane = [right_x1, height, right_x2, height // 2]
        else:
            right_lane = None

        return {'left': left_lane, 'right': right_lane}


class EnhancedModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Initialize models
        self.model = ModelTrainer(MODEL_NAME, NUM_CLASSES)
        self.lane_detector = LaneDetector()

        # Model weights for combining predictions
        self.base_weight = 0.7
        self.lane_weight = 0.3

        # Load the model
        self.load_model()

        # Initialize prediction smoothing
        self.prev_predictions = []
        self.smoothing_window = 3

    def load_model(self):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.model.to(self.device)
            self.model.model.eval()
            print("Model loaded successfully")

            # Load additional metadata if available
            self.epoch = checkpoint.get('epoch', 0)
            self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
            print(f"Model checkpoint from epoch {self.epoch} with accuracy {self.best_accuracy:.2f}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _get_lane_adjustment(self, lane_info):
        """Convert lane detection into steering adjustment"""
        if not lane_info or (lane_info['left'] is None and lane_info['right'] is None):
            return 0.0

        frame_center = 224 / 2  # Using resized frame width

        if lane_info['left'] and lane_info['right']:
            # Both lanes detected - use center point
            left_x = lane_info['left'][0]
            right_x = lane_info['right'][0]
            center = (left_x + right_x) / 2
            return (frame_center - center) / frame_center
        elif lane_info['left']:
            # Only left lane - try to maintain distance
            return (frame_center - (lane_info['left'][0] + 100)) / frame_center
        elif lane_info['right']:
            # Only right lane - try to maintain distance
            return (frame_center - (lane_info['right'][0] - 100)) / frame_center

        return 0.0

    def _smooth_predictions(self, prediction):
        """Apply temporal smoothing to predictions"""
        self.prev_predictions.append(prediction)
        if len(self.prev_predictions) > self.smoothing_window:
            self.prev_predictions.pop(0)

        # Average the predictions
        return torch.mean(torch.stack(self.prev_predictions), dim=0)

    def predict(self, frame):
        # Get base model prediction
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            base_output = self.model.model(input_tensor)
            base_prediction = base_output > PREDICTION_THRESHOLD

        # Get lane detection
        lane_info = self.lane_detector.detect(frame)
        lane_adjustment = self._get_lane_adjustment(lane_info)

        # Combine predictions
        final_prediction = base_prediction.clone()
        if len(final_prediction.shape) > 1 and final_prediction.shape[1] > 0:
            # Adjust steering (assuming first output is steering)
            steering_adjustment = torch.tensor(lane_adjustment).to(self.device)
            final_prediction[0, 0] = (self.base_weight * base_prediction[0, 0] +
                                      self.lane_weight * steering_adjustment)

        # Apply temporal smoothing
        smoothed_prediction = self._smooth_predictions(final_prediction)

        return {
            'prediction': smoothed_prediction,
            'lane_info': lane_info,
            'raw_output': base_output,
            'confidence': torch.sigmoid(base_output)
        }