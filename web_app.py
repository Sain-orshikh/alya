"""
Flask web interface for Anime Mood Detector.
Simple image upload to detect emotion and display anime face.
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from flask import Flask, render_template, request, jsonify

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.inference import EmotionPredictor
from src.emotion_mapper import AnimeEmotionMapper

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize models
try:
    predictor = EmotionPredictor()
    mapper = AnimeEmotionMapper()
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    predictor = None
    mapper = None


def convert_image_to_base64(image_path):
    """Convert image file to base64 string for embedding in HTML."""
    if not os.path.exists(image_path):
        return None
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def process_uploaded_image(file):
    """
    Process uploaded image file.
    
    Returns:
        tuple: (emotion, confidence, all_probs_dict, anime_image_base64)
    """
    try:
        # Read image
        img_data = file.read()
        image = Image.open(BytesIO(img_data)).convert('RGB')
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = predictor.detect_faces(image_cv)
        
        if not faces:
            return None, None, None, None, "No face detected"
        
        # Get first face
        x1, y1, x2, y2 = faces[0]
        face_crop = image_cv[y1:y2, x1:x2]
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        
        # Predict emotion
        emotion, confidence, probs_dict = predictor.predict_emotion(face_pil)
        
        # Get anime face
        anime_path = mapper.get_anime_path(emotion)
        anime_base64 = convert_image_to_base64(anime_path) if anime_path else None
        
        return emotion, confidence, probs_dict, anime_base64, None
        
    except Exception as e:
        return None, None, None, None, f"Error processing image: {str(e)}"


@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for emotion prediction."""
    if not request.files or 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    if predictor is None or mapper is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Process image
    emotion, confidence, probs_dict, anime_base64, error = process_uploaded_image(file)
    
    if error:
        return jsonify({'error': error}), 400
    
    if emotion is None:
        return jsonify({'error': 'Could not detect emotion'}), 400
    
    # Format probabilities for display
    emotion_probs = []
    if probs_dict:
        for emo, prob in sorted(probs_dict.items(), key=lambda x: x[1], reverse=True):
            emotion_probs.append({
                'emotion': emo.capitalize(),
                'probability': float(prob),
                'percentage': f"{prob*100:.1f}%"
            })
    
    response = {
        'emotion': emotion.capitalize(),
        'confidence': float(confidence),
        'confidence_percent': f"{confidence*100:.1f}%",
        'probabilities': emotion_probs,
        'anime_image': anime_base64
    }
    
    return jsonify(response), 200


@app.route('/home')
def home():
    """Get home anime image."""
    try:
        home_path = mapper.get_home_image_path()
        anime_base64 = convert_image_to_base64(home_path) if home_path else None
        return jsonify({'image': anime_base64}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
