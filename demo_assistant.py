#!/usr/bin/env python3
"""
Demo script for My Personal Voice Assistant
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from core.config import Config
from core.assistant import PersonalAssistant


def demo_voice_cloning(assistant):
    """Demonstrate voice cloning capabilities."""
    print("\n🎤 Voice Cloning Demo")
    print("-" * 30)
    
    # Show available voices
    voices = assistant.get_available_voices()
    print(f"Available voices: {voices}")
    
    # Test voice synthesis (if voices available)
    if voices:
        print(f"Using voice: {voices[0]}")
        assistant.set_active_voice(voices[0])
        
        test_text = "Hello! This is a demonstration of voice cloning technology."
        print(f"Generating speech: '{test_text}'")
        
        # In a real implementation, this would play the audio
        audio_data = assistant.synthesize_speech(test_text, voices[0])
        if audio_data:
            print("✅ Audio synthesized successfully!")
        else:
            print("⚠️ Audio synthesis failed (this is normal without proper TTS setup)")
    else:
        print("No cloned voices available. You can add voices through the web interface.")


def demo_vision_processing(assistant):
    """Demonstrate vision processing capabilities."""
    print("\n👁️ Vision Processing Demo")
    print("-" * 30)
    
    # Test camera
    print("Starting camera...")
    camera_started = assistant.vision_processor.start_camera()
    
    if camera_started:
        print("✅ Camera started successfully")
        
        # Start processing
        print("Starting vision processing...")
        processing_started = assistant.vision_processor.start_processing()
        
        if processing_started:
            print("✅ Vision processing started")
            
            # Wait a moment for processing
            print("Processing for 3 seconds...")
            time.sleep(3)
            
            # Get current vision state
            vision_state = assistant.vision_processor.get_current_vision_state()
            detections = vision_state.get('detections', [])
            relationships = vision_state.get('relationships', [])
            
            print(f"Detected objects: {len(detections)}")
            for detection in detections[:3]:  # Show first 3
                print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")
            
            print(f"Spatial relationships: {len(relationships)}")
            for rel in relationships[:2]:  # Show first 2
                print(f"  - {rel['object1']} is {rel['relationship']} {rel['object2']}")
            
            # Stop processing
            assistant.vision_processor.stop_processing()
            assistant.vision_processor.stop_camera()
            print("✅ Vision processing stopped")
        else:
            print("❌ Failed to start vision processing")
            assistant.vision_processor.stop_camera()
    else:
        print("⚠️ Camera not available (this is normal if no camera is connected)")


def demo_conversation(assistant):
    """Demonstrate conversation capabilities."""
    print("\n💬 Conversation Demo")
    print("-" * 30)
    
    # Test different conversation scenarios
    test_scenarios = [
        "Hello, how are you today?",
        "What's the weather like?",
        "Can you tell me a joke?",
        "What do you see in the environment?",
        "How can you help me?"
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nScenario {i}: {scenario}")
        print("Assistant:", end=" ")
        
        try:
            response = assistant.process_text_input(scenario)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(1)  # Brief pause between scenarios


def demo_personality(assistant):
    """Demonstrate personality management."""
    print("\n🎭 Personality Demo")
    print("-" * 30)
    
    # Show current personality
    personality = assistant.personality_manager.get_personality()
    print(f"Current personality: {personality.get('name', 'Unknown')}")
    
    # Test different personality presets
    presets = assistant.personality_manager.get_available_presets()
    print(f"Available presets: {presets}")
    
    for preset in presets[:2]:  # Test first 2 presets
        print(f"\nTesting '{preset}' personality:")
        assistant.personality_manager.set_personality_preset(preset)
        
        test_message = "Tell me about yourself"
        response = assistant.process_text_input(test_message)
        print(f"Response: {response[:100]}...")


def demo_vision_integration(assistant):
    """Demonstrate vision integration with conversation."""
    print("\n🔍 Vision Integration Demo")
    print("-" * 30)
    
    # Start vision processing
    if assistant.vision_processor.start_camera():
        assistant.vision_processor.start_processing()
        time.sleep(2)  # Let it process for a moment
        
        # Test vision-aware conversation
        vision_questions = [
            "What objects do you see?",
            "Are there any books in the room?",
            "What's on the table?",
            "Can you describe the spatial relationships?"
        ]
        
        for question in vision_questions:
            print(f"\nQ: {question}")
            print("A:", end=" ")
            
            try:
                response = assistant.process_text_input(question)
                print(response)
            except Exception as e:
                print(f"Error: {e}")
            
            time.sleep(1)
        
        # Stop vision
        assistant.vision_processor.stop_processing()
        assistant.vision_processor.stop_camera()
    else:
        print("⚠️ Camera not available for vision integration demo")


def main():
    """Run the demo."""
    print("🎬 My Personal Voice Assistant Demo")
    print("=" * 50)
    
    try:
        # Initialize assistant
        print("Initializing assistant...")
        config = Config()
        assistant = PersonalAssistant(config)
        print("✅ Assistant initialized successfully!")
        
        # Run demos
        demos = [
            ("Voice Cloning", lambda: demo_voice_cloning(assistant)),
            ("Vision Processing", lambda: demo_vision_processing(assistant)),
            ("Conversation", lambda: demo_conversation(assistant)),
            ("Personality", lambda: demo_personality(assistant)),
            ("Vision Integration", lambda: demo_vision_integration(assistant))
        ]
        
        for demo_name, demo_func in demos:
            try:
                demo_func()
                print(f"✅ {demo_name} demo completed")
            except Exception as e:
                print(f"❌ {demo_name} demo failed: {e}")
            
            print("\n" + "="*50)
        
        print("🎉 Demo completed!")
        print("\nTo use the full assistant:")
        print("1. Run: python src/main.py --mode web")
        print("2. Open the web interface")
        print("3. Upload voice samples and start conversations!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 