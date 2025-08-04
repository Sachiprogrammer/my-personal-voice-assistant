#!/usr/bin/env python3
"""
Test script for My Personal Voice Assistant
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from core.config import Config
from core.assistant import PersonalAssistant


def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        config = Config()
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False


def test_assistant_initialization():
    """Test assistant initialization."""
    print("Testing assistant initialization...")
    try:
        config = Config()
        assistant = PersonalAssistant(config)
        print("✅ Assistant initialized successfully")
        return assistant
    except Exception as e:
        print(f"❌ Assistant initialization failed: {e}")
        return None


def test_voice_components(assistant):
    """Test voice-related components."""
    print("Testing voice components...")
    
    try:
        # Test voice cloner
        if assistant.voice_cloner:
            print("✅ Voice cloner initialized")
        else:
            print("❌ Voice cloner not initialized")
            return False
        
        # Test speech recognizer
        if assistant.speech_recognizer:
            print("✅ Speech recognizer initialized")
        else:
            print("❌ Speech recognizer not initialized")
            return False
        
        # Test speech synthesizer
        if assistant.speech_synthesizer:
            print("✅ Speech synthesizer initialized")
        else:
            print("❌ Speech synthesizer not initialized")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Voice components test failed: {e}")
        return False


def test_vision_components(assistant):
    """Test vision-related components."""
    print("Testing vision components...")
    
    try:
        # Test object detector
        if assistant.object_detector:
            print("✅ Object detector initialized")
        else:
            print("❌ Object detector not initialized")
            return False
        
        # Test vision processor
        if assistant.vision_processor:
            print("✅ Vision processor initialized")
        else:
            print("❌ Vision processor not initialized")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Vision components test failed: {e}")
        return False


def test_conversation_components(assistant):
    """Test conversation-related components."""
    print("Testing conversation components...")
    
    try:
        # Test conversation manager
        if assistant.conversation_manager:
            print("✅ Conversation manager initialized")
        else:
            print("❌ Conversation manager not initialized")
            return False
        
        # Test personality manager
        if assistant.personality_manager:
            print("✅ Personality manager initialized")
        else:
            print("❌ Personality manager not initialized")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Conversation components test failed: {e}")
        return False


def test_basic_functionality(assistant):
    """Test basic assistant functionality."""
    print("Testing basic functionality...")
    
    try:
        # Test text processing
        test_message = "Hello, how are you?"
        response = assistant.process_text_input(test_message)
        print(f"✅ Text processing: {response[:50]}...")
        
        # Test personality
        personality = assistant.personality_manager.get_personality()
        print(f"✅ Personality loaded: {personality.get('name', 'Unknown')}")
        
        # Test available voices
        voices = assistant.get_available_voices()
        print(f"✅ Available voices: {len(voices)}")
        
        # Test status
        status = assistant.get_status()
        print(f"✅ Assistant status: {status.get('initialized', False)}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_vision_functionality(assistant):
    """Test vision functionality (if camera available)."""
    print("Testing vision functionality...")
    
    try:
        # Test camera start (this might fail if no camera)
        camera_started = assistant.vision_processor.start_camera()
        if camera_started:
            print("✅ Camera started successfully")
            assistant.vision_processor.stop_camera()
        else:
            print("⚠️ Camera not available (this is normal if no camera)")
        
        # Test vision state
        vision_state = assistant.vision_processor.get_current_vision_state()
        print(f"✅ Vision state retrieved: {vision_state.get('camera_active', False)}")
        
        return True
    except Exception as e:
        print(f"❌ Vision functionality test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing My Personal Voice Assistant")
    print("=" * 50)
    
    # Test configuration
    if not test_configuration():
        print("❌ Configuration test failed. Exiting.")
        return False
    
    # Test assistant initialization
    assistant = test_assistant_initialization()
    if not assistant:
        print("❌ Assistant initialization failed. Exiting.")
        return False
    
    # Test components
    tests = [
        ("Voice Components", lambda: test_voice_components(assistant)),
        ("Vision Components", lambda: test_vision_components(assistant)),
        ("Conversation Components", lambda: test_conversation_components(assistant)),
        ("Basic Functionality", lambda: test_basic_functionality(assistant)),
        ("Vision Functionality", lambda: test_vision_functionality(assistant))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The assistant is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 