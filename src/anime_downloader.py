"""
Anime Faces Downloader & Setup Helper.
Downloads sample anime character images from web sources or creates placeholders.
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import config


class AnimeDownloader:
    """Download and setup anime faces."""
    
    def __init__(self, anime_dir=config.ANIME_FACES_DIR):
        self.anime_dir = anime_dir
        self.emotions = config.EMOTIONS.values()
        
        # Create emotion directories
        for emotion in self.emotions:
            os.makedirs(os.path.join(anime_dir, emotion), exist_ok=True)
    
    def create_placeholder(self, emotion, index=0):
        """
        Create a placeholder image for testing.
        In production, replace with actual anime character images.
        """
        size = (400, 400)
        
        # Color map for emotions
        color_map = {
            'happy': (255, 200, 0),      # Yellow
            'sad': (100, 150, 255),      # Blue
            'angry': (255, 100, 100),    # Red
            'surprised': (200, 100, 255), # Purple
            'neutral': (180, 180, 180),  # Gray
            'fear': (255, 150, 0),       # Orange
            'disgust': (150, 200, 100),  # Green
        }
        
        color = color_map.get(emotion, (200, 200, 200))
        
        # Create image
        image = Image.new('RGB', size, color)
        draw = ImageDraw.Draw(image)
        
        # Draw simple face
        # Head
        head_bbox = [50, 50, 350, 350]
        draw.ellipse(head_bbox, outline=(0, 0, 0), width=3)
        
        # Eyes based on emotion
        if emotion == 'happy':
            # Eyes with smile
            draw.ellipse([120, 130, 160, 160], fill=(0, 0, 0))
            draw.ellipse([240, 130, 280, 160], fill=(0, 0, 0))
            draw.arc([140, 200, 260, 280], 0, 180, fill=(0, 0, 0), width=3)
        
        elif emotion == 'sad':
            # Sad eyes
            draw.ellipse([120, 140, 160, 170], fill=(0, 0, 0))
            draw.ellipse([240, 140, 280, 170], fill=(0, 0, 0))
            draw.arc([140, 260, 260, 200], 180, 360, fill=(0, 0, 0), width=3)
        
        elif emotion == 'angry':
            # Angry eyes
            draw.line([120, 130, 160, 160], fill=(0, 0, 0), width=3)
            draw.line([240, 130, 280, 160], fill=(0, 0, 0), width=3)
            draw.line([120, 160, 160, 130], fill=(0, 0, 0), width=3)
            draw.line([240, 160, 280, 130], fill=(0, 0, 0), width=3)
        
        elif emotion == 'surprised':
            # Surprised eyes (big)
            draw.ellipse([110, 120, 170, 180], fill=(0, 0, 0))
            draw.ellipse([230, 120, 290, 180], fill=(0, 0, 0))
            draw.ellipse([135, 220, 265, 280], fill=(0, 0, 0))  # O mouth
        
        elif emotion == 'neutral':
            # Neutral face
            draw.ellipse([120, 140, 160, 170], fill=(0, 0, 0))
            draw.ellipse([240, 140, 280, 170], fill=(0, 0, 0))
            draw.line([140, 240, 260, 240], fill=(0, 0, 0), width=3)
        
        elif emotion == 'fear':
            # Fearful face
            draw.ellipse([110, 120, 170, 180], fill=(0, 0, 0))
            draw.ellipse([230, 120, 290, 180], fill=(0, 0, 0))
            draw.polygon([200, 220, 180, 280, 220, 280], fill=(0, 0, 0))
        
        elif emotion == 'disgust':
            # Disgusted face
            draw.ellipse([120, 140, 160, 170], fill=(0, 0, 0))
            draw.ellipse([240, 140, 280, 170], fill=(0, 0, 0))
            draw.line([140, 230, 260, 270], fill=(0, 0, 0), width=3)
            draw.line([260, 230, 140, 270], fill=(0, 0, 0), width=3)
        
        # Add emotion text
        try:
            draw.text((200, 370), emotion.upper(), fill=(0, 0, 0), anchor="mm")
        except:
            pass
        
        # Save
        emotion_dir = os.path.join(self.anime_dir, emotion)
        filename = f"{emotion}_{index}.png"
        filepath = os.path.join(emotion_dir, filename)
        image.save(filepath)
        
        return filepath
    
    def create_all_placeholders(self):
        """Create placeholder images for all emotions."""
        print("\nCreating placeholder anime faces for testing...")
        print("-" * 70)
        
        for emotion in self.emotions:
            paths = []
            # Create 2 placeholder images per emotion
            for i in range(2):
                path = self.create_placeholder(emotion, index=i)
                paths.append(path)
                print(f"✓ Created: {path}")
        
        print("\n" + "="*70)
        print("Placeholder images created successfully!")
        print("="*70)
        print("""
These are TEST placeholders. For better results:

1. Replace with actual anime character images:
   - Search: "anime happy expression", "anime sad face", etc.
   - Sites: Danbooru, Pixiv, Tenor, Google Images
   - Download high-quality PNG images

2. Quality tips:
   - 400x400 pixels or larger
   - Clear, centered faces
   - PNG with transparency preferred
   - 1-3 images per emotion recommended

3. Save to data/anime_faces/<emotion>/

Placeholder images will work for testing, but real anime
faces will look much better in the app!
        """)
    
    def download_from_web(self, emotion, url):
        """
        Download image from URL.
        
        Args:
            emotion (str): Emotion name
            url (str): Image URL
        
        Returns:
            bool: Success status
        """
        try:
            print(f"Downloading {emotion} from {url}...")
            
            emotion_dir = os.path.join(self.anime_dir, emotion)
            filename = f"{emotion}_{len(os.listdir(emotion_dir))}.png"
            filepath = os.path.join(emotion_dir, filename)
            
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ Saved: {filepath}")
            return True
        
        except Exception as e:
            print(f"✗ Failed to download: {e}")
            return False


def print_download_guide():
    """Print guide for manually downloading anime faces."""
    print("\n" + "="*70)
    print("HOW TO GET ANIME CHARACTER IMAGES")
    print("="*70)
    print("""
RECOMMENDED SOURCES:

1. Danbooru (Free, requires account)
   - Search for: "happy anime girl", "sad anime boy", etc.
   - Filter by: "highres", PNG format
   - Save images to data/anime_faces/<emotion>/

2. Pixiv (Free/Premium)
   - Search for: "アニメ 笑顔", "アニメ 悲しい", etc.
   - Or: "anime smile", "anime cry", etc.
   - Filter by: Rating, Resolution

3. Tenor (Free, no account needed)
   - Search for: "anime happy", "anime sad", etc.
   - Right-click → Save image as
   - Save to data/anime_faces/<emotion>/

4. Unsplash/Pexels (Free)
   - Search for: "anime emotions"
   - May have limited anime-specific content

SEARCH TERMS BY EMOTION:

Happy:    "happy anime", "joyful anime", "smiling anime"
Sad:      "sad anime", "crying anime", "melancholic anime"
Angry:    "angry anime", "furious anime", "mad anime"
Surprised: "surprised anime", "shocked anime", "amazed anime"
Neutral:   "calm anime", "cool anime", "stoic anime"
Fear:      "scared anime", "frightened anime", "fearful anime"
Disgust:   "disgusted anime", "displeased anime", "annoyed anime"

DIRECTORY STRUCTURE:

data/anime_faces/
├── happy/
│   ├── happy_anime_1.png
│   ├── happy_anime_2.jpg
│   └── joyful_anime.png
├── sad/
│   ├── sad_anime_1.png
│   └── sad_anime_2.png
├── angry/
│   └── angry_anime.png
... (and so on)

TIPS:

- Minimum 1-2 images per emotion
- Recommended 3-5 for better variety
- Use PNG for transparency
- 400x400 pixels or larger
- Make sure faces are clear and centered
- Avoid copyrighted/NSFW content
    """)


def main():
    """Main function."""
    print("\n" + "="*70)
    print("ANIME FACES SETUP")
    print("="*70)
    
    downloader = AnimeDownloader()
    
    print("""
Options:
1. Create PLACEHOLDER images (for testing)
2. Manual download guide
3. Exit
    """)
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == '1':
        downloader.create_all_placeholders()
    
    elif choice == '2':
        print_download_guide()
    
    else:
        print("Exiting...")
        return
    
    print("\n" + "="*70)
    print("Remember: Add anime character images to data/anime_faces/<emotion>/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
