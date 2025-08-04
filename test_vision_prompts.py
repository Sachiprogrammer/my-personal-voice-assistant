#!/usr/bin/env python3
"""
Test script for voice assistant with specific vision prompts
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from core.config import Config
from core.assistant import PersonalAssistant


def test_vision_prompts(assistant, image_path):
    """Test the assistant with specific vision prompts."""
    
    print("🔍 Testing Vision Prompts with Image")
    print("=" * 50)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please place your image in the assets/images/ directory")
        return False
    
    print(f"📸 Using image: {image_path}")
    
    # Analyze the image
    print("\nAnalyzing image...")
    analysis = assistant.vision_processor.analyze_image(image_path)
    
    if "error" in analysis:
        print(f"❌ Image analysis failed: {analysis['error']}")
        return False
    
    print("✅ Image analyzed successfully!")
    
    # Test prompts
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
    
    print(f"\n🎯 Testing {len(prompts)} prompts:")
    print("-" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. User: {prompt}")
        print("Assistant:", end=" ")
        
        try:
            # Process the prompt with vision context
            response = assistant.process_text_input(prompt)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 30)
    
    return True


def test_voice_synthesis(assistant, prompts):
    """Test voice synthesis with the prompts."""
    
    print("\n🎤 Testing Voice Synthesis")
    print("=" * 50)
    
    # Get available voices
    voices = assistant.get_available_voices()
    
    if not voices:
        print("⚠️ No cloned voices available")
        print("To test voice synthesis, first clone a voice:")
        print("1. Upload an audio sample through the web interface")
        print("2. Or use the voice cloning API")
        return False
    
    print(f"Available voices: {voices}")
    selected_voice = voices[0]
    assistant.set_active_voice(selected_voice)
    
    print(f"Using voice: {selected_voice}")
    
    for i, prompt in enumerate(prompts[:3], 1):  # Test first 3 prompts
        print(f"\n{i}. Synthesizing: '{prompt[:50]}...'")
        
        try:
            # Generate response
            response = assistant.process_text_input(prompt)
            print(f"Response: {response[:100]}...")
            
            # Synthesize speech
            audio_data = assistant.synthesize_speech(response, selected_voice)
            if audio_data:
                print("✅ Audio synthesized successfully!")
                
                # Save audio file
                output_path = f"assets/audio/generated_response_{i}.wav"
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                print(f"💾 Saved to: {output_path}")
            else:
                print("⚠️ Audio synthesis failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return True


def main():
    """Main test function."""
    
    print("🎬 Voice Assistant Vision Prompt Testing")
    print("=" * 60)
    
    # Initialize assistant
    try:
        config = Config()
        assistant = PersonalAssistant(config)
        print("✅ Assistant initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {e}")
        return False
    
    # Look for image file
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_path = None
    
    # Check assets/images directory
    images_dir = Path("assets/images")
    if images_dir.exists():
        for ext in image_extensions:
            for img_file in images_dir.glob(f"*{ext}"):
                image_path = str(img_file)
                break
            if image_path:
                break
    
    if not image_path:
        print("❌ No image found in assets/images/")
        print("Please place your test image in the assets/images/ directory")
        print("Supported formats: JPG, PNG, BMP, TIFF")
        return False
    
    # Test prompts
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
    
    # Test vision prompts
    if not test_vision_prompts(assistant, image_path):
        return False
    
    # Test voice synthesis
    test_voice_synthesis(assistant, prompts)
    
    print("\n🎉 Testing completed!")
    print("\nNext steps:")
    print("1. Check the generated audio files in assets/audio/")
    print("2. Run the web interface: python src/main.py --mode web")
    print("3. Upload your voice sample for voice cloning")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 