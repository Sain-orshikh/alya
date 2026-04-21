# 🚀 Quick Start Guide

Complete setup and training workflow for Anime Mood Detector.

---

## Installation & Setup (3 Simple Steps)

### Step 1: Install Dependencies

```bash
cd c:\Users\xx\Desktop\alya
pip install -r requirements.txt
```

**Status:** ✅ Already installed (packages were in requirements.txt)

---

### Step 2: Download & Prepare Dataset

```bash
python setup_wizard.py
```

**What the wizard does:**
1. Checks Kaggle API setup
2. Downloads FER2013 (28K images)
3. Organizes into emotion folders
4. Creates anime faces directory
5. Validates everything

**If wizard fails:** See [DATASET_SETUP.md](DATASET_SETUP.md)

**Alternatives:**
```bash
# Just download dataset
python src/dataset_prep.py

# Create anime placeholders (for testing)
python src/anime_downloader.py
```

---

### Step 3: Train the Model

```bash
python -m src.train
```

**Training timeline:**
- Stage 1: ~1 hour (frozen backbone, 15 epochs)
- Stage 2: ~2-3 hours (fine-tune, 30 epochs)
- **Total: 2-4 hours on GPU, 8-12 hours on CPU**

**Output:**
- `models/emotion_detector_best.pth` - Best validation checkpoint
- `models/emotion_detector_final.pth` - Final trained model
- **Expected accuracy: 65-70% on validation set**

---

## Run the Application

Once training is complete:

```bash
streamlit run src/app.py
```

Then:
1. Open browser to http://localhost:8501
2. Choose "📹 Webcam" or "🖼️ Image Upload"
3. Capture image → See emotion + anime character

---

## Project Scripts & Tools

### Main Scripts

| Script | Purpose | Run With |
|--------|---------|----------|
| `setup_wizard.py` | Complete setup automation | `python setup_wizard.py` |
| `setup.py` | Validation & diagnostics | `python setup.py` |
| `src/train.py` | Train emotion detector | `python -m src.train` |
| `src/app.py` | Web interface | `streamlit run src/app.py` |

### Dataset & Setup Scripts

| Script | Purpose | Run With |
|--------|---------|----------|
| `src/dataset_prep.py` | Download & prepare FER2013 | `python src/dataset_prep.py` |
| `src/anime_downloader.py` | Setup anime faces | `python src/anime_downloader.py` |
| `DATASET_SETUP.md` | Detailed setup guide | Read documentation |

### Core Modules

| Module | Purpose |
|--------|---------|
| `src/model.py` | ResNet-50 emotion classifier |
| `src/dataset.py` | FER2013 data loader |
| `src/train.py` | Training pipeline |
| `src/inference.py` | Real-time detection |
| `src/emotion_mapper.py` | Emotion-to-anime mapping |
| `config.py` | Configuration & settings |

---

## Typical Workflow

```
1. python setup_wizard.py
   ↓
2. Download dataset (automated or manual)
   ↓
3. Add anime character images to data/anime_faces/
   ↓
4. python -m src.train
   ↓
5. streamlit run src/app.py
   ↓
6. 🎨 Point camera and enjoy!
```

---

## Common Commands

### Check if everything is installed

```bash
python setup.py
```

### Download just the dataset

```bash
python src/dataset_prep.py
```

### Create placeholder anime faces (for testing)

```bash
python -c "from src.anime_downloader import AnimeDownloader; AnimeDownloader().create_all_placeholders()"
```

### Validate dataset

```bash
python -c "from src.dataset_prep import FER2013Downloader; FER2013Downloader().verify_dataset()"
```

### Test model inference

```python
from src.inference import EmotionPredictor

predictor = EmotionPredictor()
result = predictor.predict_from_image('path/to/image.jpg')
print(result)
```

### Run webcam detection

```python
from src.inference import EmotionPredictor

predictor = EmotionPredictor()
predictor.run_webcam(camera_id=0)  # Press 'q' to quit
```

---

## Troubleshooting

### Dataset download fails

See [DATASET_SETUP.md](DATASET_SETUP.md) → Troubleshooting section

### Import errors

```bash
python setup.py
```

Then check error messages and install missing packages.

### Training is too slow

- **GPU not detected?** Check: `python -c "import torch; print(torch.cuda.is_available())"`
- **Want to reduce time?** Edit `config.py`:
  ```python
  STAGE1_EPOCHS = 5   # Reduce from 15
  STAGE2_EPOCHS = 10  # Reduce from 30
  STAGE1_BATCH_SIZE = 16  # Reduce from 32
  ```

### App won't show anime faces

1. Check anime images are in `data/anime_faces/<emotion>/`
2. Supported formats: PNG, JPG, GIF, BMP
3. Or create placeholders: `python src/anime_downloader.py` → Option 1

---

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Complete project documentation |
| [DATASET_SETUP.md](DATASET_SETUP.md) | Detailed dataset setup guide |
| [QUICKSTART.md](QUICKSTART.md) | This file - quick commands |
| [config.py](config.py) | All configuration options |

---

## Architecture Overview

```
Input (Webcam/Image)
    ↓
Face Detection (MediaPipe)
    ↓
Preprocessing (48→224px, normalize)
    ↓
ResNet-50 Model (2048→7 emotions)
    ↓
Emotion Classification (7 classes)
    ↓
Anime Mapping (emotion → anime face)
    ↓
Display (Streamlit UI)
```

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| Training time | 2-4 hrs (GPU) / 8-12 hrs (CPU) |
| Validation accuracy | 65-70% |
| Inference speed | 50-100 FPS (GPU) / 10-20 FPS (CPU) |
| Model size | ~120 MB |
| Emotions detected | 7 (angry, disgust, fear, happy, sad, surprise, neutral) |

---

## What's Included

✅ Complete PyTorch implementation
✅ ResNet-50 transfer learning
✅ Two-stage fine-tuning strategy
✅ Real-time face detection (MediaPipe)
✅ Streamlit web interface
✅ Data augmentation pipeline
✅ Model checkpointing & early stopping
✅ Comprehensive documentation
✅ Automated setup scripts
✅ Validation utilities

---

## Next Steps

1. **Install & Setup**
   ```bash
   pip install -r requirements.txt
   python setup_wizard.py
   ```

2. **Add Anime Faces** (optional for now)
   - Download from Danbooru/Pixiv
   - Or use placeholders: `python src/anime_downloader.py` → Option 1

3. **Train**
   ```bash
   python -m src.train
   ```

4. **Run App**
   ```bash
   streamlit run src/app.py
   ```

5. **Have Fun!** 🎨

---

## Support & Help

- Check [README.md](README.md) for detailed docs
- See [DATASET_SETUP.md](DATASET_SETUP.md) for data issues
- Run `python setup.py` for diagnostics
- Check `config.py` for configuration options

---

**Happy emotion detecting! May your anime faces be cute! 🎨✨**
