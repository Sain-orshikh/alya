"""
FER2013 Dataset Download and Preparation Script.
Automates downloading from Kaggle and organizing into emotion directories.

Prerequisites:
- Kaggle API setup: https://www.kaggle.com/settings/account
- Run: kaggle api-token (creates ~/.kaggle/kaggle.json)
"""

import os
import shutil
import zipfile
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import config


class FER2013Downloader:
    """Download and prepare FER2013 dataset."""
    
    def __init__(self, dataset_dir=config.FER2013_DIR):
        self.dataset_dir = dataset_dir
        self.raw_data_path = os.path.join(os.path.dirname(dataset_dir), 'fer2013_raw')
        self.csv_file = os.path.join(self.raw_data_path, 'fer2013.csv')
        
        # Emotion mapping
        self.emotions = config.EMOTIONS
        self.emotion_dirs = {v: os.path.join(dataset_dir, v) for v in self.emotions.values()}
    
    def download_kaggle(self):
        """Download FER2013 from Kaggle using Kaggle API."""
        print("\n" + "="*60)
        print("DOWNLOADING FER2013 FROM KAGGLE")
        print("="*60)
        
        # Check if Kaggle API is installed and configured
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            print("✗ Kaggle API not installed. Install with:")
            print("  pip install kaggle")
            return False
        
        # Check if Kaggle credentials exist
        kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
        if not os.path.exists(kaggle_json):
            print("✗ Kaggle API credentials not found at ~/.kaggle/kaggle.json")
            print("\nSetup Kaggle API:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Click 'Create New API Token' (downloads kaggle.json)")
            print("3. Move kaggle.json to ~/.kaggle/")
            print("4. Run: chmod 600 ~/.kaggle/kaggle.json  (on Linux/Mac)")
            return False
        
        try:
            # Authenticate
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            print("Downloading FER2013 dataset... (this may take a few minutes)")
            api.dataset_download_files('msambare/fer2013', path=self.raw_data_path, unzip=True)
            print("✓ Download complete!")
            return True
        
        except Exception as e:
            print(f"✗ Error downloading dataset: {e}")
            return False
    
    def download_alternative(self):
        """
        Alternative: Download from cloud storage if Kaggle API fails.
        (Placeholder for alternative sources)
        """
        print("\n" + "="*60)
        print("ALTERNATIVE DOWNLOAD METHOD")
        print("="*60)
        print("""
If Kaggle API doesn't work, you can:

1. Download manually from Kaggle:
   - Go to https://www.kaggle.com/datasets/msambare/fer2013
   - Click 'Download' (requires Kaggle account)
   - Extract to: data/fer2013_raw/

2. After manual download, run:
   python -c "from src.dataset_prep import FER2013Downloader; FER2013Downloader().prepare_dataset()"
        """)
        return False
    
    def prepare_dataset(self):
        """
        Parse CSV and organize images into emotion directories.
        CSV format: emotion,pixels,Usage
        """
        print("\n" + "="*60)
        print("PREPARING DATASET")
        print("="*60)
        
        if not os.path.exists(self.csv_file):
            print(f"✗ CSV file not found: {self.csv_file}")
            print("Make sure FER2013 is downloaded and extracted.")
            return False
        
        # Create emotion directories
        for emotion, emotion_dir in self.emotion_dirs.items():
            os.makedirs(emotion_dir, exist_ok=True)
        
        print(f"\nReading CSV file: {self.csv_file}")
        
        try:
            with open(self.csv_file, 'r') as f:
                csv_reader = csv.reader(f)
                
                # Skip header
                next(csv_reader)
                
                # Count total rows for progress bar
                total_rows = sum(1 for _ in open(self.csv_file)) - 1
                f.seek(0)
                next(csv_reader)
                
                # Process each row
                emotion_counts = {v: 0 for v in self.emotions.values()}
                
                with tqdm(total=total_rows, desc="Organizing images") as pbar:
                    for row in csv_reader:
                        if len(row) < 3:
                            continue
                        
                        try:
                            emotion_idx = int(row[0])
                            pixels = row[1].split()
                            usage = row[2].strip()
                            
                            # Only use Training data for now
                            if usage != 'Training':
                                pbar.update(1)
                                continue
                            
                            emotion_name = self.emotions.get(emotion_idx)
                            if not emotion_name:
                                pbar.update(1)
                                continue
                            
                            # Reconstruct image
                            image_array = np.array(pixels, dtype=np.uint8).reshape(48, 48)
                            image = Image.fromarray(image_array, mode='L')  # Grayscale
                            
                            # Save image
                            image_count = emotion_counts[emotion_name]
                            filename = f"{emotion_name}_{image_count:05d}.png"
                            image_path = os.path.join(self.emotion_dirs[emotion_name], filename)
                            image.save(image_path)
                            
                            emotion_counts[emotion_name] += 1
                        
                        except Exception as e:
                            print(f"Error processing row: {e}")
                        
                        pbar.update(1)
            
            # Print statistics
            print("\n" + "="*60)
            print("DATASET STATISTICS")
            print("="*60)
            total = 0
            for emotion in sorted(self.emotions.values()):
                count = emotion_counts[emotion]
                total += count
                print(f"{emotion.capitalize():12} : {count:6,} images")
            print("-" * 40)
            print(f"{'TOTAL':12} : {total:6,} images")
            print("="*60)
            
            return True
        
        except Exception as e:
            print(f"✗ Error preparing dataset: {e}")
            return False
    
    def split_into_sets(self):
        """
        Split data into train/val/test (70/15/15).
        This is done in dataset.py, but we can prepare files here too.
        """
        print("\nDataset split will be done automatically during training:")
        print("  Train: 70% of data")
        print("  Val:   15% of data")
        print("  Test:  15% of data")
    
    def cleanup_raw(self):
        """Remove raw CSV file after processing."""
        if os.path.exists(self.raw_data_path):
            print(f"\nCleaning up raw data: {self.raw_data_path}")
            shutil.rmtree(self.raw_data_path)
            print("✓ Cleanup complete")
    
    def verify_dataset(self):
        """Verify dataset integrity."""
        print("\n" + "="*60)
        print("VERIFYING DATASET")
        print("="*60)
        
        all_good = True
        total_images = 0
        
        for emotion in sorted(self.emotions.values()):
            emotion_dir = self.emotion_dirs[emotion]
            
            if not os.path.exists(emotion_dir):
                print(f"✗ Missing directory: {emotion_dir}")
                all_good = False
                continue
            
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            count = len(image_files)
            total_images += count
            
            if count == 0:
                print(f"✗ {emotion:12} : NO IMAGES")
                all_good = False
            else:
                print(f"✓ {emotion:12} : {count:6,} images")
        
        print("-" * 40)
        print(f"✓ Total        : {total_images:6,} images")
        
        if all_good and total_images > 0:
            print("\n✓ Dataset verification PASSED!")
            return True
        else:
            print("\n✗ Dataset verification FAILED!")
            return False


def main():
    """Main setup flow."""
    print("\n" + "="*70)
    print("FER2013 DATASET DOWNLOAD & PREPARATION")
    print("="*70)
    
    downloader = FER2013Downloader()
    
    # Check if dataset already exists
    if downloader.verify_dataset():
        print("\n✓ Dataset already prepared! Skipping download.")
        return True
    
    print("\nDataset not found. Starting download process...\n")
    
    # Try Kaggle API
    print("Method 1: Using Kaggle API (automated)")
    print("-" * 70)
    success = downloader.download_kaggle()
    
    if not success:
        print("\nMethod 2: Manual download")
        print("-" * 70)
        downloader.download_alternative()
        
        # Ask user if they've downloaded manually
        response = input("\nHave you downloaded and extracted FER2013? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Please download the dataset manually first.")
            return False
    
    # Prepare dataset
    print("\nStep 2: Organizing images into emotion directories...")
    if not downloader.prepare_dataset():
        print("Failed to prepare dataset.")
        return False
    
    # Verify
    if not downloader.verify_dataset():
        print("Dataset verification failed!")
        return False
    
    # Cleanup
    print("\nCleaning up temporary files...")
    downloader.cleanup_raw()
    
    print("\n" + "="*70)
    print("✓ DATASET SETUP COMPLETE!")
    print("="*70)
    print("\nYou're ready to train! Run:")
    print("  python -m src.train")
    print("="*70 + "\n")
    
    return True


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
