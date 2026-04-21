"""
Emotion to Anime Face Mapping.
Maps detected emotions to pre-curated anime character faces.
"""

import os
import random
from pathlib import Path
import config


class AnimeEmotionMapper:
    """Maps emotions to anime faces for display."""
    
    def __init__(self, anime_dir=config.ANIME_FACES_DIR):
        """
        Initialize mapper.
        
        Args:
            anime_dir (str): Directory containing anime image files
        """
        self.anime_dir = anime_dir
        self.emotion_to_anime = config.EMOTION_TO_ANIME
        self.home_image = config.ALYA_HOME_IMAGE
        self.loaded_images = {}
        
        self._verify_anime_directory()
    
    def _verify_anime_directory(self):
        """Verify that anime faces directory and files exist."""
        if not os.path.exists(self.anime_dir):
            print(f"Warning: Anime faces directory not found at {self.anime_dir}")
            return
        
        # Check for emotion image files
        found_emotions = 0
        for emotion, filename in self.emotion_to_anime.items():
            filepath = os.path.join(self.anime_dir, filename)
            if os.path.exists(filepath):
                print(f"✓ {emotion}: {filename}")
                found_emotions += 1
            else:
                print(f"✗ {emotion}: {filename} NOT FOUND")
        
        # Check for home image
        home_path = os.path.join(self.anime_dir, self.home_image)
        if os.path.exists(home_path):
            print(f"✓ home: {self.home_image}")
        else:
            print(f"✗ home: {self.home_image} NOT FOUND")
        
        print(f"\nTotal emotions: {found_emotions}/7")
    
    def _get_image_files(self, directory):
        """Get all image files from directory."""
        if not os.path.exists(directory):
            return []
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        images = [f for f in os.listdir(directory) 
                 if f.lower().endswith(image_extensions)]
        return images
    
    def get_anime_path(self, emotion):
        """
        Get path to anime face for given emotion.
        
        Args:
            emotion (str): Emotion name (e.g., 'happy', 'sad')
        
        Returns:
            str: Path to anime face image, or None if not available
        """
        # Get configured anime file for this emotion
        if emotion not in self.emotion_to_anime:
            print(f"Warning: No anime mapping for emotion '{emotion}'")
            return None
        
        anime_file = self.emotion_to_anime[emotion]
        if not anime_file:
            return None
        
        # Construct full path
        full_path = os.path.join(self.anime_dir, anime_file)
        
        if os.path.exists(full_path):
            return full_path
        else:
            print(f"Warning: Anime image not found at {full_path}")
            return None
    
    def get_home_image_path(self):
        """
        Get path to home/idle anime image.
        
        Returns:
            str: Path to home image, or None if not available
        """
        home_path = os.path.join(self.anime_dir, self.home_image)
        if os.path.exists(home_path):
            return home_path
        print(f"Warning: Home image not found at {home_path}")
        return None
    
    def load_home_image(self):
        """
        Load home/idle anime image.
        Caches image to avoid repeated disk reads.
        
        Returns:
            PIL.Image: Home anime image, or None if not available
        """
        # Check cache first
        if 'home' in self.loaded_images:
            return self.loaded_images['home']
        
        # Get path to home image
        home_path = self.get_home_image_path()
        if not home_path:
            return None
        
        # Load image
        try:
            from PIL import Image
            image = Image.open(home_path)
            # Cache it
            self.loaded_images['home'] = image
            return image
        except Exception as e:
            print(f"Error loading home image from {home_path}: {e}")
            return None
    
    def load_anime_image(self, emotion):
        """
        Load anime face image for given emotion.
        Caches images to avoid repeated disk reads.
        
        Args:
            emotion (str): Emotion name
        
        Returns:
            PIL.Image or np.ndarray: Anime face image, or None if not available
        """
        # Check cache first
        if emotion in self.loaded_images:
            return self.loaded_images[emotion]
        
        # Get path to anime image
        anime_path = self.get_anime_path(emotion)
        if not anime_path:
            return None
        
        # Load image
        try:
            from PIL import Image
            image = Image.open(anime_path)
            # Cache it
            self.loaded_images[emotion] = image
            return image
        except Exception as e:
            print(f"Error loading image from {anime_path}: {e}")
            return None
    
    def get_intensity_emoji(self, confidence):
        """
        Get emoji/text representation of confidence level.
        
        Args:
            confidence (float): Confidence score (0-1)
        
        Returns:
            str: Intensity indicator
        """
        if confidence >= 0.8:
            return "★★★★★ (Very Strong)"
        elif confidence >= 0.6:
            return "★★★★ (Strong)"
        elif confidence >= 0.5:
            return "★★★ (Moderate)"
        else:
            return "★★ (Weak)"
    
    def get_anime_description(self, emotion, confidence):
        """
        Get description text for anime character based on emotion and confidence.
        
        Args:
            emotion (str): Emotion name
            confidence (float): Confidence score
        
        Returns:
            str: Description text
        """
        intensity = self.get_intensity_emoji(confidence)
        descriptions = {
            'happy': f"Happy anime! {intensity}",
            'sad': f"Sad anime... {intensity}",
            'angry': f"Angry anime! {intensity}",
            'surprised': f"Surprised anime! {intensity}",
            'neutral': f"Neutral anime. {intensity}",
            'fear': f"Fearful anime! {intensity}",
            'disgust': f"Disgusted anime... {intensity}",
        }
        return descriptions.get(emotion, f"Anime ({emotion}) {intensity}")


def setup_anime_directory(anime_dir=config.ANIME_FACES_DIR):
    """
    Create anime faces directory structure and print setup instructions.
    
    Args:
        anime_dir (str): Root directory for anime faces
    """
    os.makedirs(anime_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ANIME FACES SETUP INSTRUCTIONS")
    print("="*60)
    print(f"\nCreate the following directory structure in: {anime_dir}\n")
    
    for emotion in config.EMOTIONS.values():
        emotion_dir = os.path.join(anime_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        print(f"✓ {emotion_dir}/")
    
    print("\nPlace anime face images in each emotion directory:")
    print("- Name format: happy_anime.png, happy_anime2.png, etc.")
    print("- Supported formats: .jpg, .jpeg, .png, .bmp, .gif")
    print("- Recommended size: 400x400 pixels (will be resized)")
    print("- At least 1-2 images per emotion recommended")
    print("\nExample structure:")
    print("""
    data/anime_faces/
    ├── happy/
    │   ├── happy_anime.png
    │   ├── joyful_anime.png
    ├── sad/
    │   ├── sad_anime.png
    │   ├── melancholic_anime.png
    ├── angry/
    │   └── angry_anime.png
    ... (and so on for each emotion)
    """)
    print("="*60 + "\n")


if __name__ == '__main__':
    # Setup anime directory
    setup_anime_directory()
    
    # Test mapper
    print("Testing AnimeEmotionMapper...")
    mapper = AnimeEmotionMapper()
    
    # Test getting anime path for each emotion
    for emotion in config.EMOTIONS.values():
        path = mapper.get_anime_path(emotion)
        desc = mapper.get_anime_description(emotion, 0.75)
        print(f"{emotion}: {desc}")
