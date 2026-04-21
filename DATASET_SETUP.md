# 📥 Dataset & Setup Guide

Complete guide for downloading FER2013 dataset and setting up anime faces.

## Quick Start (Automated)

Run the setup wizard to automate everything:

```bash
python setup_wizard.py
```

This will guide you through:
1. Kaggle API setup
2. Automatic dataset download
3. Anime faces directory setup
4. Validation

---

## Step-by-Step Manual Setup

### Step 1: Setup Kaggle API

The easiest way to download FER2013 is using Kaggle API.

#### 1a. Create Kaggle Account & API Token

1. Go to https://www.kaggle.com/settings/account
2. Scroll down to "API" section
3. Click **"Create New API Token"**
   - This downloads `kaggle.json`
4. Move `kaggle.json` to:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

#### 1b. Install Kaggle API

```bash
pip install kaggle
```

---

### Step 2: Download FER2013 Dataset

#### Option A: Automatic Download (Recommended)

```bash
python -c "from src.dataset_prep import FER2013Downloader; FER2013Downloader().download_kaggle(); FER2013Downloader().prepare_dataset()"
```

Or run directly:

```bash
python src/dataset_prep.py
```

**What happens:**
- Downloads FER2013 (~300 MB) from Kaggle
- Extracts and organizes into emotion directories
- Creates images in `data/fer2013/`
- Takes 5-15 minutes depending on internet speed

#### Option B: Manual Download

If Kaggle API doesn't work:

1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Click **"Download"** (requires Kaggle login)
3. Extract to `data/fer2013_raw/`
4. Run the preparation script:

```bash
python src/dataset_prep.py
```

### Dataset Structure

After download, you'll have:

```
data/fer2013/
├── angry/        (3,995 images)
├── disgust/      (436 images)
├── fear/         (4,097 images)
├── happy/        (8,989 images)
├── sad/          (6,077 images)
├── surprise/     (3,205 images)
└── neutral/      (6,198 images)

Total: ~28,709 training images
```

### Verify Dataset

Check that dataset is ready:

```bash
python -c "from src.dataset_prep import FER2013Downloader; FER2013Downloader().verify_dataset()"
```

Expected output:

```
✓ angry       :   3,995 images
✓ disgust     :     436 images
✓ fear        :   4,097 images
✓ happy       :   8,989 images
✓ neutral     :   6,198 images
✓ sad         :   6,077 images
✓ surprise    :   3,205 images
─────────────────────────────
✓ Total       :  28,709 images
```

---

## Step 3: Setup Anime Faces

### Option A: Create Placeholders (for testing)

Quick setup with placeholder images:

```bash
python -c "from src.anime_downloader import AnimeDownloader; AnimeDownloader().create_all_placeholders()"
```

Creates simple placeholder faces for each emotion. Good for testing the pipeline!

### Option B: Use Real Anime Characters (Recommended)

Download actual anime character images:

```bash
python src/anime_downloader.py
```

Menu will show:
- Option 1: Create placeholders
- Option 2: Manual download guide

### Anime Faces Directory Structure

```
data/anime_faces/
├── happy/
│   ├── happy_anime_1.png
│   └── joyful_anime.png
├── sad/
│   ├── sad_anime_1.png
│   └── sad_anime_2.jpg
├── angry/
│   └── angry_anime.png
├── surprised/
│   └── surprised_anime.png
├── neutral/
│   └── calm_anime.png
├── fear/
│   └── scared_anime.png
└── disgust/
    └── disgusted_anime.png
```

### Where to Find Anime Images

**Free Sites (No Account):**
- **Tenor**: https://tenor.com (search "anime happy")
- **Giphy**: https://giphy.com (search "anime emotions")
- **Pexels**: https://www.pexels.com (search "anime")

**Free Sites (Account Required):**
- **Danbooru**: https://danbooru.donmai.us
  - Search: "happy" + "smile" + "anime" + "highres"
  - Download as PNG
- **Pixiv**: https://pixiv.net
  - Search: "アニメ 笑顔" (anime + smile in Japanese)
  - Filter by rating and resolution

**Search Terms by Emotion:**

| Emotion | Search Terms |
|---------|--------------|
| Happy | "happy anime", "smiling anime girl", "joyful anime" |
| Sad | "sad anime", "crying anime", "melancholic anime boy" |
| Angry | "angry anime", "furious anime", "mad anime girl" |
| Surprised | "surprised anime", "shocked anime", "amazed anime" |
| Neutral | "cool anime", "calm anime", "stoic anime" |
| Fear | "scared anime", "frightened anime", "fearful anime" |
| Disgust | "disgusted anime", "annoyed anime", "displeased anime" |

### Image Guidelines

✓ **Good images have:**
- Clear, centered face
- High resolution (400x400+ pixels)
- PNG format with transparency (preferred)
- Obvious emotion expression

✗ **Avoid:**
- Blurry or low-res images
- Multiple characters
- Copyrighted content
- NSFW images

### Quick Manual Setup

1. **Create directories:**
   ```bash
   mkdir -p data/anime_faces/{happy,sad,angry,surprised,neutral,fear,disgust}
   ```

2. **Download images:**
   - Go to Danbooru or Pixiv
   - Search for emotion
   - Download 1-2 images per emotion
   - Save to corresponding folder

3. **Example:**
   - Download `happy_girl.png` → `data/anime_faces/happy/happy_girl.png`
   - Download `sad_boy.jpg` → `data/anime_faces/sad/sad_boy.jpg`

---

## Verification

### Full System Check

```bash
python setup.py
```

Or manually:

```python
from src.dataset_prep import FER2013Downloader
from src.emotion_mapper import AnimeEmotionMapper

# Check dataset
downloader = FER2013Downloader()
downloader.verify_dataset()

# Check anime faces
mapper = AnimeEmotionMapper()
```

---

## Troubleshooting

### SSL Error during download

**Error:** `ssl.SSLError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC]`

**Solution:** This is a network issue, not your code. Try:
```bash
pip install --upgrade certifi
python src/dataset_prep.py
```

Or download manually from Kaggle.

### Kaggle API not found

**Error:** `ModuleNotFoundError: No module named 'kaggle'`

**Solution:**
```bash
pip install kaggle
```

Then setup credentials at https://www.kaggle.com/settings/account

### CSV file not found

**Error:** `CSV file not found: data/fer2013_raw/fer2013.csv`

**Solution:**
1. Make sure you extracted the Kaggle download
2. Check folder: `data/fer2013_raw/` exists
3. Try manual download from Kaggle

### No images in emotion folders

**Error:** Emotion folders exist but are empty

**Solution:**
1. Check if CSV was extracted to `data/fer2013_raw/`
2. Run the script again: `python src/dataset_prep.py`
3. Check disk space (dataset is ~300MB)

### Anime faces not showing in app

**Error:** App runs but no anime images display

**Solution:**
1. Add images to: `data/anime_faces/<emotion>/`
2. Supported formats: PNG, JPG, GIF
3. Or create placeholders: `python src/anime_downloader.py` → Option 1

---

## Data Details

### FER2013 Dataset

| Statistic | Value |
|-----------|-------|
| Total Images | 35,887 |
| Training | 28,709 |
| Public Test | 3,589 |
| Private Test | 3,589 |
| Image Size | 48×48 pixels (grayscale) |
| Emotions | 7 basic emotions |
| Source | Automatically registered faces |

### Emotion Distribution (Training Set)

```
Happy:    8,989 (31.3%) ← Most images
Neutral:  6,198 (21.6%)
Sad:      6,077 (21.1%)
Fear:     4,097 (14.3%)
Angry:    3,995 (13.9%)
Surprise: 3,205 (11.2%)
Disgust:    436 ( 1.5%) ← Least images
```

**Note:** Imbalanced dataset! "Disgust" has much fewer images. The model may struggle with this emotion.

---

## Next Steps

After dataset and anime faces are ready:

### 1. Train the Model

```bash
python -m src.train
```

- Stage 1: 15 epochs (frozen backbone)
- Stage 2: 30 epochs (fine-tuning)
- Takes 2-4 hours on GPU, 8-12 hours on CPU
- Saves best model automatically

### 2. Run the Web App

```bash
streamlit run src/app.py
```

- Opens at http://localhost:8501
- Choose Webcam or Image Upload
- See real-time emotion detection

### 3. Test on Images

```python
from src.inference import EmotionPredictor
from src.emotion_mapper import AnimeEmotionMapper

predictor = EmotionPredictor()
mapper = AnimeEmotionMapper()

# Detect from image
result = predictor.predict_from_image('path/to/image.jpg')
for detection in result['detections']:
    emotion = detection['emotion']
    anime = mapper.load_anime_image(emotion)
    anime.show()
```

---

## FAQ

**Q: Can I use a different dataset?**
- Yes! Modify `src/dataset.py` to load your custom dataset. Just ensure consistent format and emotion labels.

**Q: What if I want more emotions (10-20)?**
- Option A: Map 7 base emotions to multiple anime faces with intensity levels
- Option B: Collect additional labeled data and retrain

**Q: Can I skip the anime faces for now?**
- Yes! Run training without anime faces. Create placeholders later: `python src/anime_downloader.py` → Option 1

**Q: How much disk space do I need?**
- FER2013: ~300 MB
- Trained model: ~120 MB
- Anime faces: 5-50 MB (depending on image count)
- Total: ~500 MB recommended

**Q: Can I use GPU?**
- Yes! Automatically detected by PyTorch. Check: `torch.cuda.is_available()`

**Q: Training is too slow on CPU**
- Consider using Google Colab (free GPU): https://colab.research.google.com
- Or reduce batch size in `config.py`

---

## Support

Need help? Check:
- [README.md](README.md) - Complete documentation
- [config.py](config.py) - Configuration options
- Issues with imports? Run: `python setup.py`

Happy training! 🎨
