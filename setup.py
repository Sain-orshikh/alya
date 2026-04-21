"""
Quick setup and validation script for the Anime Mood Detector project.
Run this to verify all modules can be imported and basic tests pass.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import config
        print("✓ config module")
        
        from src.model import create_model, count_parameters
        print("✓ src.model module")
        
        from src.dataset import FER2013Dataset, get_data_loaders
        print("✓ src.dataset module")
        
        from src.inference import EmotionPredictor
        print("✓ src.inference module")
        
        from src.emotion_mapper import AnimeEmotionMapper, setup_anime_directory
        print("✓ src.emotion_mapper module")
        
        print("\n✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        return False


def test_model_creation():
    """Test model creation and parameter counting."""
    print("\nTesting model creation...")
    try:
        from src.model import create_model, count_parameters
        import config
        
        model = create_model(freeze_backbone=True)
        params = count_parameters(model)
        
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {params['total']:,}")
        print(f"  - Trainable: {params['trainable']:,}")
        print(f"  - Non-trainable: {params['non_trainable']:,}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False


def check_directories():
    """Check that all necessary directories exist."""
    print("\nChecking directory structure...")
    
    import config
    
    required_dirs = [
        config.DATA_DIR,
        config.FER2013_DIR,
        config.ANIME_FACES_DIR,
        config.MODELS_DIR,
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}")
        all_exist = all_exist and exists
    
    return all_exist


def main():
    """Run all setup tests."""
    print("="*60)
    print("ANIME MOOD DETECTOR - SETUP VALIDATION")
    print("="*60)
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Directory Structure", check_directories),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SETUP VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        all_passed = all_passed and result
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All checks passed! Project is ready to use.")
        print("\nNext steps:")
        print("1. Download FER2013 dataset to data/fer2013/")
        print("2. Add anime face images to data/anime_faces/")
        print("3. Run training: python -m src.train")
        print("4. Start web interface: streamlit run src/app.py")
    else:
        print("\n✗ Some checks failed. Please fix issues before proceeding.")
        print("See error messages above for details.")
    
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
