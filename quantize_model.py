#!/usr/bin/env python3
"""
Quantize the emotion detector model for faster CPU inference.
Converts 32-bit float model to 8-bit integer for ~2-3x speedup.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import EmotionDetector
from config import TRAINED_MODEL_PATH

def quantize_model():
    """Load model, quantize to int8, and save."""
    
    print("🔄 Loading original model...")
    model = EmotionDetector(num_classes=7, pretrained=False, freeze_backbone=False)
    
    # Load checkpoint
    checkpoint = torch.load(TRAINED_MODEL_PATH, map_location='cpu')
    
    # Fix key mismatch: convert 'resnet.' prefix to 'backbone.'
    fixed_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('resnet.'):
            # Convert 'resnet.xxx' to 'backbone.xxx'
            new_key = key.replace('resnet.', 'backbone.', 1)
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value
    
    model.load_state_dict(fixed_state_dict, strict=False)
    model.eval()  # Set to evaluation mode
    
    print(f"✓ Loaded model from {TRAINED_MODEL_PATH}")
    
    # Get original model size
    original_size = os.path.getsize(TRAINED_MODEL_PATH) / (1024 * 1024)
    print(f"  Original size: {original_size:.2f} MB")
    
    # Dynamic quantization (doesn't require calibration data)
    print("\n🔄 Quantizing model (int8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},  # quantize linear and conv layers
        dtype=torch.qint8
    )
    print("✓ Quantization complete")
    
    # Save quantized model
    quantized_path = TRAINED_MODEL_PATH.replace('.pth', '_quantized.pth')
    torch.save(quantized_model.state_dict(), quantized_path)
    
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"\n✓ Saved quantized model to {quantized_path}")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Compression: {(1 - quantized_size/original_size) * 100:.1f}%")
    
    # Verify quantized model works
    print("\n🔄 Verifying quantized model...")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = quantized_model(dummy_input)
    print(f"✓ Model verification successful (output shape: {output.shape})")
    
    print("\n✅ Quantization complete!")
    print(f"\nNext steps:")
    print(f"1. In backend_api.py, change model path from:")
    print(f"   {TRAINED_MODEL_PATH}")
    print(f"   to:")
    print(f"   {quantized_path}")
    print(f"2. Push to GitHub and redeploy on Render")
    print(f"3. Expected speedup: 2-3x faster inference")

if __name__ == '__main__':
    quantize_model()
