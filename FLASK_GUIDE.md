# 🎨 Flask Web App Guide

## Quick Start

### 1. Install Flask
```bash
pip install Flask>=2.3.0
```

Or install all dependencies including Flask:
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
python web_app.py
```

The app will start at **http://localhost:5000**

### 3. How to Use

1. **Upload an image**: Click or drag-and-drop a photo with your face
2. **Wait for analysis**: The model will detect your face and analyze emotion
3. **See your Anime Mood**: View the corresponding anime character that matches your emotion
4. **Check confidence**: See the confidence score and breakdown of all emotions

## Features

✨ **7 Emotion Detection**: Angry, Disgusted, Fearful, Happy, Sad, Surprised, Neutral

📊 **Emotion Probabilities**: See full breakdown of all emotion predictions

🎨 **Anime Character Mapping**: Each emotion displays a unique anime character

📱 **Responsive Design**: Works on desktop, tablet, and mobile

🚀 **Fast Processing**: Real-time emotion detection using ResNet-50

## Technical Details

### Architecture
- **Model**: ResNet-50 with emotion classification head
- **Face Detection**: MediaPipe for robust face detection
- **Framework**: Flask for web server, vanilla JS for frontend
- **Image Processing**: OpenCV, PIL, PyTorch

### API Endpoints

#### `POST /api/predict`
Upload an image and get emotion prediction.

**Request:**
```
Content-Type: multipart/form-data
Body: image (file)
```

**Response:**
```json
{
  "emotion": "Happy",
  "confidence": 0.95,
  "confidence_percent": "95.0%",
  "probabilities": [
    {"emotion": "Happy", "probability": 0.95, "percentage": "95.0%"},
    {"emotion": "Neutral", "probability": 0.03, "percentage": "3.0%"},
    ...
  ],
  "anime_image": "base64_encoded_image_data"
}
```

#### `GET /home`
Get the home/idle anime image.

**Response:**
```json
{
  "image": "base64_encoded_image_data"
}
```

## File Structure

```
web_app.py                 # Main Flask application
templates/
  └── index.html          # Main HTML interface
static/
  └── style.css           # Styling with gradient backgrounds
data/anime_faces/         # Anime character images
```

## Customization

### Change Anime Faces
Edit `config.py` to map emotions to different anime images:

```python
EMOTION_TO_ANIME = {
    'happy': 'your_image.jpg',
    'sad': 'another_image.jpg',
    # ... other emotions
}
```

### Styling
Modify `static/style.css` to customize colors, fonts, and layout.

### Model Path
Update `config.py` if your trained model is in a different location:

```python
TRAINED_MODEL_PATH = 'path/to/your/model.pth'
```

## Deployment

### Local Network
Share with friends on your network:
```bash
python web_app.py  # Then access from their device at: http://your_ip:5000
```

### Cloud Deployment Options

**Free Options:**
- **Railway**: Supports Python, easy Flask deployment (free tier available)
- **Render**: Simple Flask deployment (spins down when idle on free tier)
- **Replit**: Code online, deploy immediately
- **Heroku**: (free tier recently removed, but still cheapest option)

**To deploy on Railway/Render:**
1. Push code to GitHub
2. Connect repository to platform
3. Platform auto-deploys when you push

### Production Considerations
For production deployment, use a proper WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app
```

## Troubleshooting

### "No face detected"
- Ensure your face is clearly visible
- Try a different angle or lighting
- Make sure the image is not too zoomed in/out

### "Model not found"
- Check `config.py` and verify `TRAINED_MODEL_PATH`
- Ensure your `.pth` file is in the correct location

### "Anime image not found"
- Verify anime face images are in `data/anime_faces/`
- Check filenames match `EMOTION_TO_ANIME` in `config.py`

### Performance Issues
- Reduce image size before upload (recommended: <5MB)
- Run on a machine with GPU for faster processing
- Consider caching predictions

## Tips for Friends

1. **Best results**: Well-lit face photos
2. **Exaggerate emotions**: Bigger expressions = higher confidence
3. **Different angles**: Try frontal and slightly angled shots
4. **Multiple attempts**: Different photos may give different results

Enjoy! 🎨✨
