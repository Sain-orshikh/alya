"""
Inference module for real-time emotion detection.
Handles face detection and emotion prediction from webcam or images.
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
import config
from src.model import create_model


class EmotionPredictor:
    """Real-time emotion prediction from faces."""
    
    def __init__(self, model_path=config.TRAINED_MODEL_PATH, device=config.DEVICE):
        """
        Initialize emotion predictor.
        
        Args:
            model_path (str): Path to trained model
            device: torch device
        """
        self.device = device
        self.model = create_model(freeze_backbone=False, pretrained=False)
        
        # Load trained weights
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Fix key mismatch: convert 'resnet.' prefix to 'backbone.'
                # and 'resnet.' keys to 'backbone.' keys
                fixed_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('resnet.'):
                        # Convert 'resnet.xxx' to 'backbone.xxx'
                        new_key = key.replace('resnet.', 'backbone.', 1)
                        fixed_state_dict[new_key] = value
                    else:
                        fixed_state_dict[key] = value
                
                self.model.load_state_dict(fixed_state_dict, strict=False)
                print(f"✓ Model loaded from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
        else:
            print(f"Warning: Model not found at {model_path}")
        
        self.model.eval()
        self.model.to(device)
        
        # Initialize face detector using Haar Cascade (OpenCV built-in)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        
        if self.face_detector.empty():
            print("Warning: Haar Cascade classifier not found")
        
        # Preprocessing transform
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
    
    def detect_faces(self, image):
        """
        Detect faces in image using Haar Cascade.
        
        Args:
            image (np.ndarray): Input image (BGR format from OpenCV)
        
        Returns:
            list: List of detected face bounding boxes [(x1, y1, x2, y2), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        result = []
        for (x, y, w, h) in faces:
            # Add margin around face
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            result.append((x1, y1, x2, y2))
        
        return result
    
    def predict_emotion(self, face_image):
        """
        Predict emotion for a face image.
        
        Args:
            face_image (np.ndarray or PIL.Image): Face image
        
        Returns:
            tuple: (emotion_name, confidence, all_probs)
        """
        # Convert to PIL if needed
        if isinstance(face_image, np.ndarray):
            face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        
        # Preprocess
        face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(face_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)
        
        emotion_idx = predicted_idx.item()
        emotion_name = config.EMOTIONS[emotion_idx]
        confidence = confidence.item()
        
        # Get all probabilities
        all_probs = {
            config.EMOTIONS[i]: probs[0, i].item()
            for i in range(config.NUM_EMOTIONS)
        }
        
        return emotion_name, confidence, all_probs
    
    def process_frame(self, frame):
        """
        Process a frame: detect faces and predict emotions.
        
        Args:
            frame (np.ndarray): Input frame (BGR)
        
        Returns:
            tuple: (processed_frame, detections)
                - processed_frame: Frame with drawn bboxes and emotions
                - detections: List of {face_bbox, emotion, confidence}
        """
        detections = []
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Process each face
        for (x1, y1, x2, y2) in faces:
            face_roi = frame[y1:y2, x1:x2]
            
            # Skip very small faces
            if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                continue
            
            # Predict emotion
            emotion, confidence, all_probs = self.predict_emotion(face_roi)
            
            # Skip low confidence predictions
            if confidence < config.CONFIDENCE_THRESHOLD:
                continue
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'emotion': emotion,
                'confidence': confidence,
                'all_probs': all_probs
            })
            
            # Draw on frame
            frame = self._draw_detection(frame, x1, y1, x2, y2, emotion, confidence)
        
        return frame, detections
    
    def _draw_detection(self, frame, x1, y1, x2, y2, emotion, confidence):
        """Draw bounding box and emotion label on frame."""
        # Color map for emotions
        color_map = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'surprised': (0, 255, 255), # Cyan
            'neutral': (128, 128, 128), # Gray
            'fear': (255, 0, 255),      # Magenta
            'disgust': (0, 165, 255),  # Orange
        }
        
        color = color_map.get(emotion, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{emotion.capitalize()} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        label_y = max(y1 - 10, label_size[1])
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0] + 5, label_y + 5), color, -1)
        cv2.putText(frame, label, (x1 + 5, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run_webcam(self, camera_id=0, window_name="Emotion Detector"):
        """
        Run real-time emotion detection from webcam.
        
        Args:
            camera_id (int): Camera device ID
            window_name (str): Window title
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print(f"Starting webcam (press 'q' to quit)...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame, detections = self.process_frame(frame)
                
                # Display detections info
                if detections:
                    info_text = f"Detected {len(detections)} face(s): " + \
                               ", ".join([f"{d['emotion']}" for d in detections])
                    cv2.putText(frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Break on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def predict_from_image(self, image_path):
        """
        Predict emotion from image file.
        
        Args:
            image_path (str): Path to image
        
        Returns:
            dict: Processed frame and detections
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        frame, detections = self.process_frame(frame)
        
        return {
            'frame': frame,
            'detections': detections,
            'image_path': image_path
        }


import os


def test_inference():
    """Test inference on sample image."""
    print("Testing EmotionPredictor...")
    
    predictor = EmotionPredictor()
    
    # Create dummy image for testing
    print("Creating test image...")
    dummy_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Process frame (will not detect faces since it's random noise)
    processed_frame, detections = predictor.process_frame(dummy_image)
    print(f"Detections: {len(detections)}")
    print("Inference test passed!")


if __name__ == '__main__':
    test_inference()
