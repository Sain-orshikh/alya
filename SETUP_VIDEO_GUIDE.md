# 🎨 Anime Mood Detector - React + FastAPI Setup Guide

## Overview

This is a real-time emotion detection system with:
- **Frontend**: React 18 with Vite (fast dev server)
- **Backend**: FastAPI with async processing
- **Architecture**: Two-page React app (Home + Real-time Predict)

## System Architecture

### Pages

1. **Homepage** (`/`)
   - Beautiful landing page
   - Feature cards explaining the system
   - "Start Detecting" button to begin
   - How it works section

2. **Predict Page** (`/predict`)
   - Webcam feed as main background
   - Anime character display (centered)
   - 5-second capture intervals for emotion detection
   - Loading effect while processing
   - Real-time emotion results with confidence score
   - All 7 emotion probabilities breakdown
   - Back button to return to homepage

### Capture Flow

```
Every 5 seconds:
1. Capture frame from webcam
2. Send to backend API (/api/predict)
3. Backend detects face and predicts emotion
4. Returns: emotion, confidence, probabilities, anime image
5. Frontend displays results with anime character
6. Show loading spinner during processing
```

## Prerequisites

- **Python 3.10+** with PyTorch installed
- **Node.js 16+** and npm
- **Model file**: `models/emotion_detector_best.pth`
- **Anime images**: `data/anime_faces/` directory with emotion-mapped images

## Installation

### 1. Backend Setup

**Option A: Using existing Python environment**

```bash
# In project root (c:\Users\xx\Desktop\alya)
pip install -r requirements.txt
```

**Option B: Create new virtual environment**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Then install requirements
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Verify installation
npm list react react-router-dom axios
```

## Running the Application

### Start Backend Server

```bash
# From project root (in a terminal)
python backend_api.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Start Frontend Dev Server

```bash
# From frontend directory (in a NEW terminal)
npm run dev
```

Expected output:
```
  VITE v4.3.0  ready in 200 ms

  ➜  Local:   http://localhost:3000/
  ➜  press h to show help
```

### Access the Application

Open browser and navigate to:
```
http://localhost:3000
```

You should see:
1. Beautiful homepage with feature cards
2. "Start Detecting" button
3. Click button to go to `/predict` page

## Using the Predict Page

1. **Allow Camera Access**: Browser will prompt for camera permission
2. **Wait for Face Detection**: Show your face to the camera
3. **5-Second Captures**: Every 5 seconds, a frame is captured
4. **Loading Effect**: Brief loading spinner appears during processing
5. **See Results**: Emotion, confidence score, and anime character display
6. **Probabilities**: See breakdown of all 7 emotions

## Project Structure

```
frontend/
├── package.json              # Dependencies (React, Vite, Axios, React Router)
├── vite.config.js           # Vite dev server config with API proxy
├── public/
│   └── index.html           # HTML entry point
└── src/
    ├── main.jsx             # React root
    ├── App.jsx              # Router setup
    ├── App.css              # Global styles
    ├── index.css            # Base CSS
    └── pages/
        ├── HomePage.jsx     # Home page component
        ├── HomePage.css     # Home page styling
        ├── PredictPage.jsx  # Predict page component
        └── PredictPage.css  # Predict page styling

backend_api.py              # FastAPI server (4 endpoints)
config.py                   # Configuration & emotion mappings
requirements.txt            # Python dependencies
```

## API Endpoints

### 1. Health Check
```
GET /
Response: {"status": "ok"}
```

### 2. Predict Emotion
```
POST /api/predict
Content-Type: multipart/form-data
Body: file (image)

Response:
{
  "emotion": "happy",
  "confidence": 0.95,
  "anime_image": "base64_encoded_image",
  "probabilities": [
    {"emotion": "happy", "probability": 0.95, "percentage": "95%"},
    {"emotion": "sad", "probability": 0.03, "percentage": "3%"},
    ...
  ]
}
```

### 3. Home/Idle Image
```
GET /api/home-image
Response:
{
  "image": "base64_encoded_image"
}
```

### 4. List Emotions
```
GET /api/emotions
Response:
{
  "emotions": ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"]
}
```

## Customization

### Change Capture Interval

Edit `frontend/src/pages/PredictPage.jsx`:

```javascript
const CAPTURE_INTERVAL = 5000 // Change to desired milliseconds
```

### Change Anime Characters

1. Replace images in `data/anime_faces/`
2. Update filenames to match emotions in `config.py`:

```python
EMOTION_TO_ANIME = {
    'happy': 'alya-happy.jpg',
    'sad': 'alya-sad.jpg',
    # ... etc
}
```

### Modify Colors & Styling

Edit the respective CSS files:
- `frontend/src/pages/HomePage.css` - Home page design
- `frontend/src/pages/PredictPage.css` - Predict page design

Key color variables in CSS:
```css
--primary: #667eea
--secondary: #764ba2
--danger: #ff6b6b
--success: #4ade80
```

## Deployment

### Backend Deployment (Railway, Render, Heroku)

1. Create account on [Railway](https://railway.app) or [Render](https://render.com)
2. Connect GitHub repository
3. Set start command:
   ```
   python backend_api.py
   ```
4. Set environment variables if needed
5. Deploy

### Frontend Deployment (Vercel, Netlify)

1. Create account on [Vercel](https://vercel.com) or [Netlify](https://netlify.com)
2. Connect GitHub repository
3. Build command:
   ```
   npm run build
   ```
4. Output directory: `dist`
5. Update `REACT_APP_API_BASE` in `frontend/src/pages/PredictPage.jsx` to point to production backend

## Troubleshooting

### Frontend won't connect to backend
- Ensure backend is running on `http://localhost:8000`
- Check CORS is enabled in `backend_api.py`
- Verify proxy in `frontend/vite.config.js` points to correct backend URL

### Camera permission denied
- Browser needs HTTPS for camera access (except localhost)
- Try different browser if permission is cached
- Clear browser cache and site data

### Model loading error
- Verify `models/emotion_detector_best.pth` exists
- Check model path in `config.py`
- Ensure PyTorch is installed: `pip install torch torchvision`

### Anime images not displaying
- Verify images exist in `data/anime_faces/`
- Check filenames match `config.py` EMOTION_TO_ANIME dictionary
- Ensure image formats are supported (jpg, png)

### Slow emotion detection
- Reduce image quality in backend (currently 0.85)
- Increase capture interval from 5 seconds to higher
- Consider GPU acceleration if available

## Performance Tips

1. **GPU Acceleration**: Install CUDA for PyTorch for faster inference
2. **Image Compression**: Reduce frame quality if bandwidth is limited
3. **Batch Processing**: For multiple faces, could be optimized further
4. **Caching**: Anime images are cached in frontend after first load

## Technologies Used

**Frontend:**
- React 18 - UI library
- React Router DOM 6 - Page navigation
- Vite 4 - Lightning-fast build tool
- Axios - HTTP client
- Modern CSS3 - Responsive design

**Backend:**
- FastAPI - High-performance async web framework
- Uvicorn - ASGI server
- PyTorch - Deep learning inference
- OpenCV - Haar Cascade face detection
- Pillow - Image processing

## License

This project uses emotion detection models and anime character representations. Ensure you have proper rights to use all assets.

## Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all dependencies are installed: `pip list`
3. Check backend console for error messages
4. Check browser console (F12) for frontend errors

---

**Created**: April 2026
**Architecture**: React + FastAPI with Real-time Video Processing
