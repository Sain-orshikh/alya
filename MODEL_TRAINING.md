# Anime Mood Detector - Model Training & System Guide

## System Overview

The Anime Mood Detector uses a ResNet-50 deep learning model trained on the FER2013 emotion dataset to detect 7 basic emotions from facial expressions and map them to anime character responses.

### Architecture

```
Input Image (Face)
    ↓
Face Detection (OpenCV Haar Cascade)
    ↓
ResNet-50 Backbone (trained on FER2013)
    ↓
Classification Head (7 emotions)
    ↓
Emotion Mapping → Anime Image Response
    ↓
Output JSON (emotion, confidence, probabilities, anime image)
```

## How It Works

### 1. Face Detection
- Uses OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`)
- Detects faces in the input image
- Crops the face region for emotion detection

### 2. Emotion Prediction
- ResNet-50 backbone trained on FER2013 dataset
- 7 emotion classes: `['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']`
- Outputs confidence scores for each emotion
- Returns the highest confidence emotion

### 3. Anime Mapping
- Maps detected emotions to anime character images
- Located in `data/anime_faces/` directory
- Images named as: `alya-{emotion}.jpg`
- Returns image as base64 in JSON response

## Model Files

- **Model**: `models/emotion_detector_best.pth` (60MB)
- **Config**: `config.py` - Contains emotion labels and image paths
- **Inference**: `src/inference.py` - EmotionPredictor class

## Backend API

### Endpoint: `POST /api/predict`

**Request:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.95,
  "probabilities": {
    "angry": 0.01,
    "disgust": 0.0,
    "fear": 0.02,
    "happy": 0.95,
    "sad": 0.01,
    "surprise": 0.01,
    "neutral": 0.0
  },
  "anime_image": "data:image/jpeg;base64,..."
}
```

## Continuing Training from Current Model

### Prerequisites
```bash
pip install -r requirements.txt
```

### Loading the Pre-trained Model

The current model is saved with state_dict keys that have a `resnet.` prefix. When loading:

```python
from src.model import EmotionDetector
from config import TRAINED_MODEL_PATH
import torch

# Load model
model = EmotionDetector(num_emotions=7)
checkpoint = torch.load(TRAINED_MODEL_PATH)

# Convert state_dict keys if needed
if any(k.startswith('resnet.') for k in checkpoint.keys()):
    checkpoint = {
        k.replace('resnet.', 'backbone.'): v 
        for k, v in checkpoint.items()
    }

model.load_state_dict(checkpoint)
model.train()  # Set to training mode
```

### Fine-tuning on New Data

```python
from src.dataset import FER2013Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

# Load dataset
train_dataset = FER2013Dataset(split='train', augment=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Set up training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Lower LR for fine-tuning
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save updated model
torch.save(model.state_dict(), TRAINED_MODEL_PATH)
print("Model fine-tuned and saved!")
```

### Training on Custom Emotion Dataset

If you want to train on a custom dataset with your own emotion labels:

```python
from src.model import EmotionDetector
from torch.utils.data import DataLoader, Dataset
import torch

class CustomEmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Use it
custom_dataset = CustomEmotionDataset(image_paths, labels, transform=your_transform)
train_loader = DataLoader(custom_dataset, batch_size=32)

# Then follow the fine-tuning steps above
```

## File Structure

```
.
├── backend_api.py           # FastAPI server
├── config.py                # Configuration (emotion labels, paths)
├── models/
│   └── emotion_detector_best.pth   # Pre-trained model (60MB)
├── data/
│   └── anime_faces/         # Anime response images
├── src/
│   ├── inference.py         # Emotion prediction logic
│   ├── model.py             # ResNet-50 model definition
│   ├── dataset.py           # Dataset loading
│   ├── train.py             # Training script
│   └── emotion_mapper.py    # Map emotions to images
├── requirements.txt         # Python dependencies
└── Procfile                 # Render deployment config
```

## Deployment

### Local Development
```bash
python backend_api.py
```

### Production (Render.com)
- Connected to GitHub repo
- Auto-deploys on push
- Environment: Python 3
- Start command: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`

## Troubleshooting

### Model not loading
- Check model path in `config.py`
- Ensure `models/emotion_detector_best.pth` exists
- Check state_dict keys if conversion is needed

### Face detection failing
- Image quality/lighting might be poor
- Face too small or partially visible
- Try with a clearer facial expression

### CORS errors on frontend
- Backend should have CORS enabled (check `backend_api.py`)
- Verify API_BASE_URL in frontend matches backend URL

## Next Steps

1. **Improve accuracy**: Train on larger dataset or newer architectures (EfficientNet, Vision Transformer)
2. **Add more emotions**: Extend beyond 7 basic emotions
3. **Multi-face detection**: Detect multiple faces in one image
4. **Real-time optimization**: Quantize model for faster inference
5. **Mobile deployment**: Convert to ONNX or TensorFlow Lite

## Resources

- **FER2013 Dataset**: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
- **PyTorch ResNet**: https://pytorch.org/vision/main/models/resnet.html
- **FastAPI**: https://fastapi.tiangolo.com/
