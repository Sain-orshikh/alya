# 🎨 Anime Mood Detector - Full Stack Setup

Complete guide for running the **video-based** emotion detector with React frontend and Python FastAPI backend.

## Architecture

```
Frontend (React + Webcam)        Backend (FastAPI)           ML Model
Port 3000                        Port 8000                   PyTorch
├─ WebcamCapture                ├─ /api/predict             ├─ ResNet-50
├─ Real-time frame capture      ├─ Face Detection           ├─ 7 Emotions
└─ Anime character display      └─ Emotion inference        └─ Anime Mapping
```

## Prerequisites

- **Python 3.10+** (with PyTorch installed)
- **Node.js 16+** and npm
- **Git** (optional)

## Setup Instructions

### 1. Backend Setup (Python FastAPI)

#### Install Dependencies

```bash
# Navigate to project root
cd path/to/alya

# Install/upgrade Python packages
pip install -r requirements.txt
```

#### Start the Backend API

```bash
# From project root directory
python backend_api.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ Backend is ready at `http://localhost:8000`
📚 API docs available at `http://localhost:8000/docs`

### 2. Frontend Setup (React)

#### Install Dependencies

```bash
cd frontend

# Install npm packages
npm install
```

#### Start the Development Server

```bash
# From frontend directory
npm run dev
# or
npm start
```

You should see:
```
VITE v4.x.x ready in xxx ms

➜  Local:   http://localhost:3000
```

✅ Frontend is ready at `http://localhost:3000`

## Usage

1. **Open browser** to `http://localhost:3000`
2. **Allow camera access** when prompted
3. **Show your face** to the webcam
4. **Watch** the anime character change based on your emotion
5. See **emotion breakdown** with confidence scores

## Project Structure

```
alya/
├── backend_api.py              # FastAPI server
├── requirements.txt            # Python dependencies
├── src/
│   ├── inference.py           # Emotion detection logic
│   ├── emotion_mapper.py       # Emotion → Anime mapping
│   ├── model.py               # ResNet-50 architecture
│   ├── train.py               # Training code
│   └── __init__.py
├── models/
│   └── emotion_detector_best.pth  # Trained model
├── data/
│   └── anime_faces/           # 7 anime character images
│       ├── alya-happy.jpg
│       ├── alya-sad.jpg
│       ├── alya-angry.jpg
│       ├── alya-surprised.jpeg
│       ├── alya-neutral.jpg
│       ├── alya-fear.jpg
│       ├── alya-disgust.jpg
│       └── alya-home.jpg
└── frontend/                  # React app
    ├── package.json
    ├── vite.config.js
    ├── public/
    │   └── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx            # Main component
        ├── index.css
        └── api/
```

## API Endpoints

### Backend API (http://localhost:8000)

**POST** `/api/predict`
- Send image frame from webcam
- Returns emotion, confidence, probabilities, anime image
- Headers: `Content-Type: multipart/form-data`

**GET** `/api/home-image`
- Get the home/idle anime character image
- Response: `{ "image": "base64_string" }`

**GET** `/api/emotions`
- List all emotion types
- Response: `{ "emotions": [...], "count": 7 }`

**GET** `/`
- Health check
- Response: `{ "status": "ok", "models_loaded": true }`

## Deployment

### Deploy Backend

**Option 1: Railway**
```bash
# Install Railway CLI
npm i -g railway

# Login and deploy
railway login
railway up
```

**Option 2: Render**
1. Push code to GitHub
2. Create new service on Render
3. Connect repository
4. Set start command: `python backend_api.py`

**Option 3: Heroku** (requires Heroku CLI)
```bash
heroku create your-app-name
git push heroku main
```

### Deploy Frontend

**Option 1: Vercel**
```bash
npm install -g vercel
cd frontend
vercel
```

**Option 2: Netlify**
```bash
npm run build
# Drag and drop 'dist' folder to Netlify
```

**Option 3: GitHub Pages**
```bash
npm run build
# Deploy 'dist' folder via GitHub Pages
```

### Important: Update API URL in Production

Edit `frontend/src/App.jsx`:
```javascript
// Change from localhost to production backend URL
const API_BASE = 'https://your-backend-url.com'
```

## Troubleshooting

### Frontend can't connect to backend
```bash
# Check backend is running on port 8000
curl http://localhost:8000

# Check frontend is running on port 3000
# Browser console should show API errors if backend is down
```

### "No face detected" repeatedly
- Ensure good lighting
- Face should be clearly visible to camera
- Try different angles
- Get closer to camera

### Model loading errors
- Verify `emotion_detector_best.pth` exists in `models/`
- Check model path in `config.py`: `TRAINED_MODEL_PATH`
- Ensure PyTorch is properly installed

### Webcam permission denied
- Check browser allows camera access
- Try a different browser
- Restart browser if permission was previously denied

### Slow frame processing
- Reduce frame capture frequency (change 500ms in App.jsx)
- Check system resources (GPU memory, CPU)
- Run backend on machine with GPU for faster inference

## Customization

### Change Emotion → Anime Mapping

Edit `config.py`:
```python
EMOTION_TO_ANIME = {
    'happy': 'your-image.jpg',
    'sad': 'another-image.jpg',
    # ... other emotions
}
```

### Modify Capture Frequency

Edit `frontend/src/App.jsx`:
```javascript
// Capture every 500ms (increase for slower processing)
captureIntervalRef.current = setInterval(captureFrame, 500)
```

### Update Styling

Edit `frontend/src/index.css` to customize colors, layout, fonts, etc.

## Performance Tips

1. **Backend**: Use GPU for faster inference
2. **Frontend**: Reduce video resolution if needed
3. **Network**: Deploy backend and frontend on same server for lower latency
4. **Model**: Current ResNet-50 is lightweight; inference ~50-100ms per frame

## Tech Stack

**Backend:**
- FastAPI - Modern Python web framework
- PyTorch - Deep learning framework
- OpenCV - Computer vision (face detection)
- Pillow - Image processing

**Frontend:**
- React 18 - UI library
- Vite - Build tool and dev server
- Axios - HTTP client

## Features

✅ Real-time webcam feed processing
✅ 7 emotion detection (angry, disgust, fear, happy, sad, surprise, neutral)
✅ Live anime character display matching emotion
✅ Confidence scores and emotion probabilities
✅ Beautiful gradient UI
✅ Responsive design (works on desktop, tablet)
✅ CORS-enabled for flexible deployment
✅ Both development and production ready

## Next Steps

1. **Customize anime faces** - Replace with your preferred characters
2. **Deploy backend** - Use Railway, Render, or Heroku
3. **Deploy frontend** - Use Vercel, Netlify, or GitHub Pages
4. **Share with friends** - Send them the deployed URL
5. **Fine-tune model** - Retrain with your own data if needed

## License

MIT

## Support

For issues or questions:
1. Check troubleshooting section
2. Review error messages in browser console
3. Check backend logs for API errors
4. Verify all dependencies are installed correctly

---

Built with ❤️ for anime fans and emotion detection enthusiasts! 🎨✨
