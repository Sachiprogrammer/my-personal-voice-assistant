#!/usr/bin/env python3
"""
Setup script for vision testing with user's image and prompts
"""

import sys
import os
import shutil
from pathlib import Path

def setup_vision_test():
    """Setup the vision testing environment."""
    
    print("🔧 Setting up Vision Testing Environment")
    print("=" * 50)
    
    # Create necessary directories
    directories = [
        "assets/images",
        "assets/audio/voice_samples",
        "assets/audio/generated_responses",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Check for image file
    print("\n📸 Looking for image file...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_found = False
    
    # Check current directory
    for ext in image_extensions:
        for img_file in Path(".").glob(f"*{ext}"):
            print(f"Found image: {img_file}")
            # Copy to assets/images
            dest_path = f"assets/images/{img_file.name}"
            shutil.copy2(img_file, dest_path)
            print(f"✅ Copied to: {dest_path}")
            image_found = True
            break
        if image_found:
            break
    
    if not image_found:
        print("⚠️ No image found in current directory")
        print("Please place your test image in the current directory or assets/images/")
        print("Supported formats: JPG, PNG, BMP, TIFF")
    
    # Create test prompts file
    prompts = [
        "Hey assistant, can you tell me where my TV is in this room?",
        "Where is my Doraemon in this picture?",
        "Can you point out where the fan is?",
        "Where's my red suitcase kept?",
        "Is there any mirror in this room? Where is it?",
        "How many beds do you see here?",
        "Which side is the floor lamp on?",
        "Can you describe what's in front of me?"
    ]
    
    with open("test_prompts.txt", "w") as f:
        for i, prompt in enumerate(prompts, 1):
            f.write(f"{i}. {prompt}\n")
    
    print("✅ Created test_prompts.txt")
    
    # Create instructions
    instructions = """
🎯 Voice Assistant Vision Testing Setup Complete!

Next Steps:

1. 📸 Image Setup:
   - Your image should be in assets/images/
   - Supported formats: JPG, PNG, BMP, TIFF

2. 🎤 Voice Cloning (Optional):
   - Upload an audio sample through the web interface
   - Or place audio files in assets/audio/voice_samples/

3. 🧪 Run Tests:
   - Test vision prompts: python test_vision_prompts.py
   - Run full demo: python demo_assistant.py
   - Web interface: python src/main.py --mode web

4. 📋 Test Prompts:
   - Check test_prompts.txt for the 8 test prompts
   - These will be used to test the assistant's vision capabilities

5. 🔧 Configuration:
   - Edit config/settings.yaml for custom settings
   - Set OPENAI_API_KEY environment variable

Ready to test! 🚀
"""
    
    with open("SETUP_INSTRUCTIONS.txt", "w") as f:
        f.write(instructions)
    
    print("✅ Created SETUP_INSTRUCTIONS.txt")
    
    print("\n🎉 Setup complete!")
    print("Check SETUP_INSTRUCTIONS.txt for next steps")


def check_requirements():
    """Check if all requirements are met."""
    
    print("🔍 Checking Requirements")
    print("=" * 30)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}")
    else:
        print(f"❌ Python version too old: {python_version.major}.{python_version.minor}")
        print("   Required: Python 3.8+")
        return False
    
    # Check for required files
    required_files = [
        "src/core/config.py",
        "src/core/assistant.py",
        "config/settings.yaml",
        "requirements.txt"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            return False
    
    # Check for environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OPENAI_API_KEY is set")
    else:
        print("⚠️ OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
    
    return True


def main():
    """Main setup function."""
    
    print("🚀 Voice Assistant Vision Testing Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return False
    
    # Setup environment
    setup_vision_test()
    
    print("\n🎯 Ready to test your voice assistant!")
    print("\nQuick start:")
    print("1. python test_vision_prompts.py")
    print("2. python src/main.py --mode web")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 