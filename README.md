# 🎨 Anime Mood Detector

Real-time emotion detection using ResNet-50 and anime character matching with PyTorch.

## 🎯 Features

- **Real-time Emotion Detection**: Detects 7 emotions (happy, sad, angry, surprised, neutral, fear, disgust)
- **Webcam Support**: Live camera feed with instant emotion detection
- **Anime Character Matching**: Maps detected emotions to cute anime characters
- **Two-Stage Fine-Tuning**: Transfer learning with ResNet-50 backbone
- **Streamlit Interface**: Beautiful web-based UI for easy interaction
- **Multi-Face Detection**: Detect and classify emotions from multiple faces simultaneously

## 📋 Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 🚀 Installation

### 1. Clone/Create Project

```bash
cd c:\Users\xx\Desktop\alya
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
# Using venv
python -m venv venv
venv\Scripts\activate

# Or using conda
conda create -n anime-mood python=3.10
conda activate anime-mood
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 Dataset Setup

### Automated Setup (Recommended)

```bash
python setup_wizard.py
```

This will guide you through the entire process:
- Setup Kaggle API credentials
- Automatically download FER2013
- Organize images into emotion directories
- Setup anime faces
- Validate everything works

### Manual Setup

For detailed step-by-step instructions, see [DATASET_SETUP.md](DATASET_SETUP.md)

#### Quick Download

```bash
python src/dataset_prep.py
```

This downloads from Kaggle and organizes into:

```
data/fer2013/
├── angry/
├── disgust/
├── fear/
├── happy/
├── sad/
├── surprise/
└── neutral/
```

**Prerequisites:** Kaggle API setup
1. Go to [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Download the dataset OR
3. Setup Kaggle API at https://www.kaggle.com/settings/account

```
data/fer2013/
├── angry/
├── disgust/
├── fear/
├── happy/
├── sad/
├── surprise/
└── neutral/
```

Each subdirectory should contain emotion-specific image files.

### Setup Anime Faces

#### Option 1: Quick Placeholders (for testing)

```bash
python src/anime_downloader.py
# Then select option 1: Create placeholders
```

Creates simple placeholder images for testing.

#### Option 2: Real Anime Characters

```bash
python src/anime_downloader.py
# Then select option 2: Manual download guide
```

Manual download guide will show you:
- Best websites to find anime characters
- Search terms for each emotion
- How to organize files

Final structure:

```
data/anime_faces/
├── happy/
│   ├── happy_anime_1.png
│   └── happy_anime_2.jpg
├── sad/
│   └── sad_anime.png
├── angry/
│   └── angry_anime.png
└── ...
```

**Where to find anime images:**
- [Danbooru](https://danbooru.donmai.us/) - Free with account
- [Pixiv](https://www.pixiv.net/) - Large collection
- [Tenor](https://tenor.com/) - GIFs and images (search "anime emotion")

**Image tips:**
- 400x400+ pixels recommended
- PNG format preferred
- 1-3 images per emotion
- Clear, centered faces

## 🎓 Training

### Two-Stage Fine-Tuning

```bash
python -m src.train
```

**Stage 1** (15 epochs): Train classification head only (frozen backbone)
- Learning rate: 0.001
- Optimizer: Adam
- Batch size: 32

**Stage 2** (30 epochs): Fine-tune entire network
- Learning rate: 0.0001 (with discriminative LR for layers)
- Optimizer: SGD
- Batch size: 32

Training will automatically save:
- `models/emotion_detector_best.pth` - Best model on validation set
- `models/emotion_detector_final.pth` - Final model after training

### Expected Results

- **Validation Accuracy**: 65-70% on FER2013
- **Training Time**: 2-4 hours on GPU, 8-12 hours on CPU
- **Model Size**: ~120 MB

## 🎮 Usage

### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run src/app.py
```

Then:
1. Open browser to `http://localhost:8501`
2. Choose **Webcam** or **Image Upload** mode
3. Capture/upload image
4. View detected emotions and matching anime characters

### Option 2: Real-time Webcam Detection

```python
from src.inference import EmotionPredictor

predictor = EmotionPredictor()
predictor.run_webcam(camera_id=0)  # Press 'q' to quit
```

### Option 3: Image Inference

```python
from src.inference import EmotionPredictor
from src.emotion_mapper import AnimeEmotionMapper

predictor = EmotionPredictor()
mapper = AnimeEmotionMapper()

result = predictor.predict_from_image('path/to/image.jpg')
frame = result['frame']
detections = result['detections']

for detection in detections:
    emotion = detection['emotion']
    confidence = detection['confidence']
    print(f"Emotion: {emotion}, Confidence: {confidence:.2%}")
    
    anime_image = mapper.load_anime_image(emotion)
    if anime_image:
        anime_image.show()
```

## 📁 Project Structure

```
alya/
├── src/
│   ├── __init__.py
│   ├── app.py                 # Streamlit web interface
│   ├── dataset.py             # FER2013 dataset loader with augmentation
│   ├── model.py               # ResNet-50 emotion classifier
│   ├── train.py               # Two-stage fine-tuning trainer
│   ├── inference.py           # Real-time emotion detection
│   └── emotion_mapper.py      # Emotion-to-anime mapping
├── data/
│   ├── fer2013/               # FER2013 dataset (download required)
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── surprise/
│   │   └── neutral/
│   └── anime_faces/           # Anime character images (user-provided)
│       ├── happy/
│       ├── sad/
│       ├── angry/
│       └── ...
├── models/                    # Trained model checkpoints
│   ├── emotion_detector_best.pth
│   └── emotion_detector_final.pth
├── config.py                  # Configuration and hyperparameters
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model
MODEL_NAME = 'resnet50'
INPUT_SIZE = 224
NUM_CLASSES = 7

# Training
STAGE1_EPOCHS = 15
STAGE1_LR = 0.001
STAGE2_EPOCHS = 30
STAGE2_LR = 0.0001

# Emotions (change to add more)
EMOTIONS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

# Anime mapping
EMOTION_TO_ANIME = {
    'happy': ['happy_anime.png', 'joyful_anime.png'],
    'sad': ['sad_anime.png'],
    # ... etc
}
```

## 🔍 Model Architecture

**ResNet-50 with Emotion Head:**

```
Input (B, 3, 224, 224)
  ↓
ResNet-50 Backbone (pre-trained ImageNet)
  ├─ Layer1: 64 channels
  ├─ Layer2: 128 channels
  ├─ Layer3: 256 channels
  └─ Layer4: 512 channels
  ↓
Global Average Pooling
  ↓
Fully Connected Layer (2048 → 7 emotions)
  ↓
Output (B, 7)
```

**Why ResNet-50?**
- Better accuracy than VGG16 (66-68% vs 60-65%)
- Lower memory footprint (25.5M vs 138M parameters)
- Faster inference (2-3x speedup)
- Residual connections for better gradient flow

## 📈 Performance

| Metric | Value |
|--------|-------|
| Train Accuracy | 85-90% |
| Validation Accuracy | 65-70% |
| Test Accuracy | 64-68% |
| Per-class Performance | Variable (happy/sad easier, disgust harder) |
| Inference Speed | ~50-100 FPS on GPU, 10-20 FPS on CPU |

## 🐛 Troubleshooting

### Model not loading
```
Error: Model not found at models/emotion_detector_final.pth
```
- Train the model first: `python -m src.train`

### No faces detected
- Check lighting conditions
- Get closer to camera
- Ensure face is centered
- Try lowering confidence threshold in app

### Anime images not displaying
- Create proper directory structure in `data/anime_faces/`
- Run: `python -c "from src.emotion_mapper import setup_anime_directory; setup_anime_directory()"`
- Add image files to emotion subdirectories

### CUDA out of memory
- Reduce batch size in config.py: `STAGE1_BATCH_SIZE = 16`
- Use CPU: `export CUDA_VISIBLE_DEVICES=""`

### Slow performance
- Use GPU: Install CUDA and cuDNN
- Reduce image resolution
- Lower inference size: `INPUT_SIZE = 192`

## 🎓 Extending the Project

### Add More Emotions (10-20)
Option A: Map existing emotions to multiple anime faces with intensity levels
```python
EMOTION_TO_ANIME = {
    'happy': ['very_happy.png', 'slightly_happy.png', 'excited.png'],
    'sad': ['very_sad.png', 'slightly_sad.png'],
    # ... etc
}
```

### Fine-tune on Custom Dataset
```python
# Modify dataset.py to load your custom emotion data
# Then run training with your data path
```

### Deploy as API
```python
# Create FastAPI server for remote inference
from fastapi import FastAPI
from src.inference import EmotionPredictor

app = FastAPI()
predictor = EmotionPredictor()

@app.post("/detect")
async def detect_emotion(image: UploadFile):
    # Process and return emotion
    pass
```

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **ResNet-50**: He et al., "Deep Residual Learning for Image Recognition" (2015)
- **FER2013 Dataset**: Pierre Carrier, Aaron Courville, et al.
- **MediaPipe**: Google's face detection framework
- **PyTorch**: Facebook AI Research

## 📧 Contact & Support

For issues or questions, refer to the troubleshooting section above.

---

**Happy emotion detecting! 🎨🎭**
