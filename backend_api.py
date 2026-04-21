"""
FastAPI backend for Anime Mood Detector
Serves emotion detection API for React frontend
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.inference import EmotionPredictor
from src.emotion_mapper import AnimeEmotionMapper

app = FastAPI(title="Anime Mood Detector API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Convert image file to base64 string."""
    if not os.path.exists(image_path):
        return None
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Anime Mood Detector API",
        "models_loaded": predictor is not None and mapper is not None
    }


@app.post("/api/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded image frame.
    
    Args:
        file: Image file from frontend
    
    Returns:
        JSON with emotion, confidence, and anime character image
    """
    if predictor is None or mapper is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Read image
        img_data = await file.read()
        
        if not img_data:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        image = Image.open(BytesIO(img_data)).convert('RGB')
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = predictor.detect_faces(image_cv)
        
        if not faces:
            return JSONResponse({
                "emotion": None,
                "confidence": 0,
                "error": "No face detected",
                "anime_image": None
            }, status_code=200)
        
        # Get first face
        x1, y1, x2, y2 = faces[0]
        face_crop = image_cv[y1:y2, x1:x2]
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        
        # Predict emotion
        emotion, confidence, probs_dict = predictor.predict_emotion(face_pil)
        
        # Get anime face
        anime_path = mapper.get_anime_path(emotion)
        anime_base64 = convert_image_to_base64(anime_path) if anime_path else None
        
        # Format probabilities
        emotion_probs = []
        if probs_dict:
            for emo, prob in sorted(probs_dict.items(), key=lambda x: x[1], reverse=True):
                emotion_probs.append({
                    'emotion': emo.capitalize(),
                    'probability': float(prob),
                    'percentage': f"{prob*100:.1f}%"
                })
        
        return {
            'emotion': emotion.capitalize(),
            'confidence': float(confidence),
            'confidence_percent': f"{confidence*100:.1f}%",
            'probabilities': emotion_probs,
            'anime_image': anime_base64
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/api/home-image")
def get_home_image():
    """Get the home/idle anime image."""
    try:
        home_path = mapper.get_home_image_path()
        anime_base64 = convert_image_to_base64(home_path) if home_path else None
        return {"image": anime_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/emotions")
def get_emotions_list():
    """Get list of all emotions."""
    return {
        "emotions": config.EMOTION_LABELS,
        "count": config.NUM_EMOTIONS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
