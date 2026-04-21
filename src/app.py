"""
Streamlit web interface for the Anime Mood Detector.
Real-time emotion detection with anime character display.
"""

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.inference import EmotionPredictor
from src.emotion_mapper import AnimeEmotionMapper, setup_anime_directory


# Page configuration
st.set_page_config(
    page_title="Anime Mood Detector",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .emotion-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }
    .confidence-bar {
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the emotion predictor model (cached)."""
    try:
        predictor = EmotionPredictor(device=config.DEVICE)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_mapper():
    """Load the anime emotion mapper (cached)."""
    return AnimeEmotionMapper()


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown("""
    <div style='text-align: center'>
        <h1>🎨 Anime Mood Detector 🎨</h1>
        <p style='font-size: 18px; color: #666;'>
            Real-time emotion detection with anime character matching
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["📹 Webcam", "🖼️ Image Upload"]
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence to display emotion"
    )
    
    # Update config
    config.CONFIDENCE_THRESHOLD = confidence_threshold
    
    # Load models
    with st.spinner("Loading models..."):
        predictor = load_predictor()
        mapper = load_mapper()
    
    if predictor is None:
        st.error("Failed to load emotion detector model. Please check the model file.")
        return
    
    # Mode: Webcam
    if mode == "📹 Webcam":
        st.subheader("Webcam Feed")
        
        # Setup anime directory
        setup_anime_directory()
        
        # Webcam input
        picture = st.camera_input("Take a picture")
        
        if picture is not None:
            # Convert to numpy array
            image = Image.open(picture)
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process frame
            processed_frame, detections = predictor.process_frame(image_np)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detection Result")
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(processed_frame_rgb, use_column_width=True)
            
            with col2:
                st.subheader("Anime Match")
                
                if detections:
                    for i, detection in enumerate(detections):
                        emotion = detection['emotion']
                        confidence = detection['confidence']
                        
                        st.write(f"**Face {i+1}:**")
                        
                        # Display emotion info
                        emotion_text = f"**{emotion.upper()}** ({confidence:.2%})"
                        st.markdown(emotion_text)
                        
                        # Display intensity
                        intensity = mapper.get_intensity_emoji(confidence)
                        st.write(f"Intensity: {intensity}")
                        
                        # Try to display anime character
                        anime_image = mapper.load_anime_image(emotion)
                        if anime_image:
                            # Resize anime image
                            anime_image = anime_image.resize((300, 300), Image.Resampling.LANCZOS)
                            st.image(anime_image, use_column_width=True, caption=f"Anime: {emotion}")
                        else:
                            st.warning(f"⚠️ Anime face image for '{emotion}' not found. Please add images to data/anime_faces/{emotion}/")
                        
                        # Show all emotion probabilities
                        with st.expander(f"Emotion Probabilities (Face {i+1})"):
                            all_probs = detection['all_probs']
                            probs_sorted = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                            for emotion_name, prob in probs_sorted:
                                st.write(f"{emotion_name.capitalize()}: {prob:.4f}")
                        
                        st.divider()
                else:
                    st.info("ℹ️ No faces detected. Please ensure good lighting and visibility.")
                    # Display home image when no face detected
                    home_image = mapper.load_home_image()
                    if home_image:
                        home_image = home_image.resize((300, 300), Image.Resampling.LANCZOS)
                        st.image(home_image, use_column_width=True, caption="Alya (Idle)")
        
        else:
            st.info("📸 Click 'Take a picture' to capture from your webcam")
            # Display home image on startup
            home_image = mapper.load_home_image()
            if home_image:
                st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
                home_image_resized = home_image.resize((300, 300), Image.Resampling.LANCZOS)
                st.image(home_image_resized, caption="Alya (Ready!)")
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Mode: Image Upload
    elif mode == "🖼️ Image Upload":
        st.subheader("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process
            with st.spinner("Detecting emotions..."):
                processed_frame, detections = predictor.process_frame(image_np)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detection Result")
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(processed_frame_rgb, use_column_width=True)
            
            with col2:
                st.subheader("Anime Matches")
                
                if detections:
                    for i, detection in enumerate(detections):
                        emotion = detection['emotion']
                        confidence = detection['confidence']
                        
                        st.write(f"### Face {i+1}")
                        
                        # Display emotion info
                        col_emotion, col_conf = st.columns([2, 1])
                        with col_emotion:
                            st.write(f"**{emotion.upper()}**")
                        with col_conf:
                            st.write(f"{confidence:.1%}")
                        
                        # Display intensity
                        intensity = mapper.get_intensity_emoji(confidence)
                        st.caption(f"Intensity: {intensity}")
                        
                        # Try to display anime character
                        anime_image = mapper.load_anime_image(emotion)
                        if anime_image:
                            anime_image = anime_image.resize((250, 250), Image.Resampling.LANCZOS)
                            st.image(anime_image, use_column_width=True, caption=emotion.capitalize())
                        else:
                            st.warning(f"⚠️ Anime image for '{emotion}' not found")
                        
                        # Show probabilities
                        with st.expander(f"All Emotions (Face {i+1})"):
                            all_probs = detection['all_probs']
                            probs_sorted = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                            for emo, prob in probs_sorted:
                                bar_length = int(prob * 20)
                                bar = "█" * bar_length + "░" * (20 - bar_length)
                                st.write(f"{emo.capitalize():12} {bar} {prob:.2%}")
                        
                        st.divider()
                else:
                    st.info("ℹ️ No faces detected in the image")
                    # Display home image when no face detected
                    home_image = mapper.load_home_image()
                    if home_image:
                        home_image = home_image.resize((300, 300), Image.Resampling.LANCZOS)
                        st.image(home_image, use_column_width=True, caption="Alya (Idle)")
        else:
            st.info("📁 Upload an image to get started")
            # Display home image on startup
            home_image = mapper.load_home_image()
            if home_image:
                st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
                home_image_resized = home_image.resize((300, 300), Image.Resampling.LANCZOS)
                st.image(home_image_resized, caption="Alya (Ready!)")
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 30px;'>
        <p>
            <strong>Anime Mood Detector</strong> • Powered by ResNet-50 & MediaPipe<br>
            Detecting emotions • Matching anime characters • Having fun! 🎨
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info sidebar
    with st.sidebar:
        st.divider()
        st.subheader("📊 Model Info")
        st.write(f"**Device:** {config.DEVICE}")
        st.write(f"**Model:** ResNet-50")
        st.write(f"**Emotions:** {config.NUM_EMOTIONS}")
        st.write(f"**Input Size:** {config.INPUT_SIZE}x{config.INPUT_SIZE}")
        
        st.divider()
        st.subheader("📖 Instructions")
        st.write("""
        1. **Select Mode:** Choose Webcam or Image Upload
        2. **Take Picture:** Capture or upload an image
        3. **Detect:** Model analyzes faces and predicts emotions
        4. **Match:** Anime characters appear based on detected mood
        
        **Tips:**
        - Good lighting helps detection
        - Look at the camera directly
        - Multiple faces are supported
        """)
        
        st.divider()
        st.subheader("⚙️ Setup Anime Faces")
        if st.button("🎨 Show Setup Instructions"):
            setup_anime_directory()
            st.success("Check the console for detailed setup instructions!")


if __name__ == '__main__':
    main()
