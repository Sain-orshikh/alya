"""
Quick-start setup script for Anime Mood Detector.
Handles dataset download, anime faces setup, and validation.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)


def print_step(step_num, text):
    """Print step."""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 70)


def setup_kaggle_api():
    """Guide user through Kaggle API setup."""
    print_step(1, "KAGGLE API SETUP")
    
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    
    if os.path.exists(kaggle_json):
        print("✓ Kaggle API credentials already configured!")
        return True
    
    print("""
To download FER2013 automatically, you need to set up Kaggle API:

1. Go to https://www.kaggle.com/settings/account
2. Scroll down and click "Create New API Token"
3. This downloads kaggle.json
4. Move it to ~/.kaggle/ folder:
   - Windows: C:\\Users\\<YourUsername>\\.kaggle\\
   - Linux/Mac: ~/.kaggle/

Then run this script again!
    """)
    
    response = input("\nHave you set up Kaggle API? (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def download_dataset():
    """Download and prepare FER2013 dataset."""
    print_step(2, "DOWNLOAD FER2013 DATASET")
    
    try:
        from src.dataset_prep import FER2013Downloader
        
        downloader = FER2013Downloader()
        
        # Check if already prepared
        if downloader.verify_dataset():
            print("\n✓ Dataset already prepared!")
            return True
        
        print("Starting automated download...")
        
        # Try Kaggle API
        if downloader.download_kaggle():
            success = downloader.prepare_dataset()
            if success:
                downloader.verify_dataset()
                downloader.cleanup_raw()
                return True
        
        # If Kaggle fails, guide manual download
        print("\n✗ Kaggle API download failed.")
        print("""
MANUAL DOWNLOAD:
1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Click "Download"
3. Extract to: data/fer2013_raw/
4. Run this script again
        """)
        
        response = input("\nHave you manually downloaded? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            downloader.prepare_dataset()
            downloader.verify_dataset()
            downloader.cleanup_raw()
            return True
        
        return False
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def setup_anime_faces():
    """Setup anime faces directory structure."""
    print_step(3, "SETUP ANIME FACES DIRECTORY")
    
    try:
        from src.emotion_mapper import setup_anime_directory
        setup_anime_directory()
        
        print("""
✓ Anime faces directory structure created!

NEXT: Add your anime character images
- Download from: Danbooru, Pixiv, or search "anime emotions"
- Save to: data/anime_faces/<emotion>/
- Name examples: happy_anime.png, sad_anime2.jpg, etc.

Example search terms:
- "anime happy expression" (for happy)
- "anime sad face" (for sad)
- "anime angry eyes" (for angry)
- etc.

You need at least 1-2 images per emotion.
        """)
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def validate_setup():
    """Validate entire setup."""
    print_step(4, "VALIDATION")
    
    try:
        print("Running validation tests...\n")
        
        # Test imports
        print("Testing module imports...")
        try:
            import config
            from src.model import create_model
            from src.dataset import FER2013Dataset
            from src.inference import EmotionPredictor
            print("✓ All modules imported successfully")
        except ImportError as e:
            print(f"✗ Import error: {e}")
            return False
        
        # Test model creation
        print("Testing model creation...")
        try:
            model = create_model(freeze_backbone=True)
            print("✓ ResNet-50 model created successfully")
        except Exception as e:
            print(f"✗ Model creation error: {e}")
            return False
        
        # Check dataset
        print("Checking dataset...")
        import os
        fer_dir = config.FER2013_DIR
        if os.path.exists(fer_dir):
            emotion_dirs = [d for d in os.listdir(fer_dir) 
                           if os.path.isdir(os.path.join(fer_dir, d))]
            total_images = sum(
                len([f for f in os.listdir(os.path.join(fer_dir, d)) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))])
                for d in emotion_dirs
            )
            print(f"✓ FER2013 dataset: {len(emotion_dirs)} emotions, {total_images:,} images")
        else:
            print(f"⚠ FER2013 dataset not found at {fer_dir}")
        
        # Check anime faces
        print("Checking anime faces...")
        anime_dir = config.ANIME_FACES_DIR
        if os.path.exists(anime_dir):
            emotion_dirs = [d for d in os.listdir(anime_dir) 
                           if os.path.isdir(os.path.join(anime_dir, d))]
            total_images = sum(
                len([f for f in os.listdir(os.path.join(anime_dir, d)) 
                    if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))])
                for d in emotion_dirs
            )
            if total_images > 0:
                print(f"✓ Anime faces: {len(emotion_dirs)} emotions, {total_images} images")
            else:
                print(f"⚠ Anime faces directory empty (add images manually)")
        
        return True
    
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False


def print_next_steps():
    """Print next steps."""
    print_header("SETUP COMPLETE!")
    
    print("""
You're ready to train! Next steps:

1. If you haven't added anime faces yet:
   - Download 1-2 anime character images per emotion
   - Place in: data/anime_faces/<emotion>/
   - Example: data/anime_faces/happy/happy_anime.png

2. Train the emotion detector:
   python -m src.train
   
   This will:
   - Load FER2013 dataset
   - Train ResNet-50 with two-stage fine-tuning
   - Save best model to models/emotion_detector_best.pth
   - Takes ~2-4 hours on GPU, 8-12 hours on CPU

3. Run the web interface:
   streamlit run src/app.py
   
   Then:
   - Open browser to http://localhost:8501
   - Choose webcam or image upload mode
   - See real-time emotion detection with anime faces!

4. Optional: Train on custom emotions (10-20)
   - Map multiple anime faces per emotion (different intensities)
   - Or collect additional training data

Questions? Check README.md for detailed documentation.
    """)


def main():
    """Main setup flow."""
    print_header("ANIME MOOD DETECTOR - SETUP WIZARD")
    
    print("""
This wizard will help you:
1. Setup Kaggle API for automatic dataset download
2. Download and prepare FER2013 dataset
3. Setup anime faces directory
4. Validate everything works
    """)
    
    input("\nPress Enter to start...")
    
    # Step 1: Kaggle API
    if not setup_kaggle_api():
        print("\n✗ Setup cancelled: Kaggle API not configured")
        print("\nManual setup:")
        print("1. Setup Kaggle API credentials")
        print("2. Run this script again")
        return False
    
    # Step 2: Download dataset
    if not download_dataset():
        print("\n✗ Dataset download failed")
        print("Please download manually from:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
        return False
    
    # Step 3: Anime faces
    setup_anime_faces()
    
    # Step 4: Validate
    if not validate_setup():
        print("\n⚠ Validation had some issues, but you can still try training")
    
    # Done!
    print_next_steps()
    
    print("\n" + "="*70)
    print("Setup wizard complete! Good luck! 🎨")
    print("="*70 + "\n")
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n✗ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
