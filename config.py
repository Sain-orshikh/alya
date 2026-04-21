"""
Configuration and constants for the Anime Mood Detector project.
"""

import os

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FER2013_DIR = os.path.join(DATA_DIR, 'fer2013')
ANIME_FACES_DIR = os.path.join(DATA_DIR, 'anime_faces')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create directories if they don't exist
os.makedirs(FER2013_DIR, exist_ok=True)
os.makedirs(ANIME_FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==================== EMOTIONS ====================
# FER2013 emotion labels (7 basic emotions)
EMOTIONS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

EMOTION_LABELS = list(EMOTIONS.values())
NUM_EMOTIONS = len(EMOTIONS)

# Reverse mapping: emotion name -> index
EMOTION_TO_IDX = {v: k for k, v in EMOTIONS.items()}

# ==================== MODEL CONFIG ====================
MODEL_NAME = 'resnet50'
PRETRAINED = True
INPUT_SIZE = 224
NUM_CLASSES = NUM_EMOTIONS

# ==================== TRAINING HYPERPARAMETERS ====================
# Stage 1: Train only head (frozen backbone)
STAGE1_EPOCHS = 15
STAGE1_BATCH_SIZE = 32
STAGE1_LR = 0.001
STAGE1_OPTIMIZER = 'adam'

# Stage 2: Fine-tune entire network
STAGE2_EPOCHS = 30
STAGE2_BATCH_SIZE = 32
STAGE2_LR = 0.0001
STAGE2_OPTIMIZER = 'sgd'
STAGE2_MOMENTUM = 0.9

# General training
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42

# ==================== DATA AUGMENTATION ====================
AUGMENTATION_CONFIG = {
    'rotation': 10,           # Random rotation ±10 degrees
    'brightness': 0.2,        # ColorJitter brightness
    'contrast': 0.2,          # ColorJitter contrast
    'horizontal_flip': True,  # Random horizontal flip
    'translate': (0.1, 0.1),  # Random translate 10%
    'cutout_p': 0.5,          # Probability of cutout
    'cutout_size': 0.2,       # Size of cutout patch
}

# Normalization (ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ==================== INFERENCE ====================
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to display emotion

# ==================== FACE DETECTION ====================
FACE_DETECTION_BACKEND = 'mediapipe'  # 'mediapipe', 'opencv', or 'dlib'
FACE_MIN_CONFIDENCE = 0.5

# ==================== ANIME FACES MAPPING ====================
# Map FER2013 emotions to anime character filenames
# Individual files in anime_faces directory (not subdirectories)
EMOTION_TO_ANIME = {
    'happy': 'alya-happy.jpg',
    'sad': 'alya-sad.jpg',
    'angry': 'alya-angry.jpg',
    'surprise': 'alya-surprised.jpeg',
    'neutral': 'alya-neutral.jpg',
    'fear': 'alya-fear.jpg',
    'disgust': 'alya-disgust.jpg',
}

# Special anime image for home/idle state (displayed when no face detected)
ALYA_HOME_IMAGE = 'alya-home.jpg'

# ==================== DEVICE ====================
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_AVAILABLE = torch.cuda.is_available()

# ==================== PATHS ====================
TRAINED_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_detector_best.pth')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_detector_best.pth')
