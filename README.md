# My Personal Voice Assistant

A comprehensive AI assistant with voice cloning, conversational AI, and spatial awareness capabilities. This assistant can see, hear, speak, and understand its environment in real-time.

## 🎯 Project Overview

This AI assistant combines three core functionalities:

1. **Voice Cloning**: Uses YourTTS to clone voices from audio samples
2. **Conversational AI**: Powered by GPT-4 for natural conversations
3. **Spatial Awareness**: Real-time object detection and spatial reasoning using YOLOv8

## ✨ Key Features

### 🎤 Voice & Audio Interaction
- **Speech-to-Text**: Real-time transcription using Whisper
- **Text-to-Speech**: Voice synthesis with cloned voices
- **Voice Cloning**: Clone any voice from audio samples
- **Natural Prosody**: Context-aware speech generation

### 🧠 Conversational AI
- **GPT-4 Integration**: Advanced language understanding
- **Vision Integration**: Seamless environment awareness
- **Personality Management**: Customizable assistant personality
- **Conversation History**: Context-aware responses

### 👁️ Vision & Spatial Awareness
- **Real-time Object Detection**: YOLOv8-based detection
- **Spatial Reasoning**: Understand object relationships
- **Live Camera Feed**: Continuous environment monitoring
- **Relationship Analysis**: "Book is on table", "Lamp is next to computer"

## 🏗️ Architecture

```
src/
├── core/
│   ├── assistant.py      # Main assistant class
│   └── config.py         # Configuration management
├── voice/
│   ├── speech_recognition.py    # Whisper-based STT
│   ├── speech_synthesis.py      # TTS with voice cloning
│   └── voice_cloner.py          # YourTTS voice cloning
├── vision/
│   ├── object_detection.py      # YOLOv8 object detection
│   └── vision_processor.py      # Real-time vision processing
├── utils/
│   ├── conversation_manager.py   # GPT-4 conversation handling
│   ├── personality_manager.py    # Personality system
│   └── logger.py                # Logging utilities
└── ui/
    └── streamlit_app.py         # Web interface
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key** (for GPT-4)
3. **Camera** (for vision features)
4. **Microphone** (for voice input)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd my-personal-voice-assistant
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export OPENAI_ORG_ID="your-openai-org-id"  # Optional
   ```

4. **Run the assistant**:
   ```bash
   # Web interface (recommended)
   python src/main.py --mode web
   
   # Command line interface
   python src/main.py --mode cli
   
   # Voice-only mode
   python src/main.py --mode voice
   ```

## 📖 Usage Guide

### Voice Cloning

1. **Prepare Audio Sample**:
   - Record 3-30 seconds of clear speech
   - Supported formats: WAV, MP3, FLAC, M4A
   - Sample rate: 16kHz+ recommended

2. **Clone Voice**:
   ```python
   # Via code
   assistant.clone_voice("path/to/audio.wav", "my_voice")
   
   # Via web interface
   # Upload audio file and click "Clone Voice"
   ```

3. **Use Cloned Voice**:
   ```python
   # Set as active voice
   assistant.set_active_voice("my_voice")
   
   # Synthesize speech
   audio = assistant.synthesize_speech("Hello, this is my cloned voice!")
   ```

### Vision & Spatial Awareness

1. **Start Camera**:
   ```python
   assistant.vision_processor.start_camera()
   assistant.vision_processor.start_processing()
   ```

2. **Ask About Environment**:
   ```
   User: "What do you see in the room?"
   Assistant: "I can see a laptop on the desk, a coffee cup next to it, and a lamp above the desk."
   
   User: "Where is the book?"
   Assistant: "I can see a book on the table to the left of the laptop."
   ```

### Personality Customization

1. **Use Presets**:
   ```python
   # Professional mode
   assistant.personality_manager.set_personality_preset("professional")
   
   # Friendly mode
   assistant.personality_manager.set_personality_preset("friendly")
   
   # Casual mode
   assistant.personality_manager.set_personality_preset("casual")
   ```

2. **Custom Personality**:
   ```python
   custom_personality = {
       "name": "Alex",
       "traits": {
           "friendly": 0.9,
           "helpful": 0.8,
           "professional": 0.6,
           "humorous": 0.7
       },
       "interests": ["technology", "music", "travel"]
   }
   assistant.personality_manager.update_personality(custom_personality)
   ```

## 🔧 Configuration

The assistant is configured via `config/settings.yaml`:

```yaml
# Voice settings
voice:
  speech_recognition:
    model: "whisper"
    language: "en"
    sample_rate: 16000
  
  speech_synthesis:
    model: "coqui-tts"
    voice_cloning:
      model: "your-tts"

# Vision settings
vision:
  camera:
    device_id: 0
    resolution: [640, 480]
    fps: 30
  
  object_detection:
    model: "yolov8"
    confidence_threshold: 0.5
    classes: ["person", "chair", "table", "book", "laptop"]

# Conversation settings
conversation:
  llm:
    provider: "openai"
    model: "gpt-4"
    temperature: 0.7
```

## 🎮 Example Scenarios

### Scenario 1: Voice Cloning
```
1. User provides audio sample of their voice
2. Assistant clones the voice using YourTTS
3. User asks: "What's the weather like?"
4. Assistant responds in the cloned voice: "The weather is sunny today!"
```

### Scenario 2: Spatial Awareness
```
1. Camera detects: laptop on desk, coffee cup next to laptop
2. User asks: "Are there any drinks in the room?"
3. Assistant responds: "Yes, I can see a coffee cup next to the laptop on the desk."
```

### Scenario 3: Personality Interaction
```
1. Set personality to "friendly"
2. User: "Tell me a joke"
3. Assistant: "Hey there! Here's a light-hearted joke for you..."
```

## 🛠️ Development

### Project Structure
```
my-personal-voice-assistant/
├── assets/
│   ├── audio/          # Voice samples
│   ├── images/         # Test images
│   └── models/         # Saved models
├── config/
│   └── settings.yaml   # Configuration
├── src/
│   ├── core/           # Core assistant logic
│   ├── voice/          # Voice processing
│   ├── vision/         # Computer vision
│   ├── utils/          # Utilities
│   └── ui/             # User interfaces
├── tests/              # Unit tests
└── docs/               # Documentation
```

### Adding New Features

1. **New Voice Model**:
   - Add model configuration in `settings.yaml`
   - Implement in `voice/speech_synthesis.py`

2. **New Vision Capability**:
   - Add to `vision/object_detection.py`
   - Update configuration classes

3. **New Personality Trait**:
   - Add to `utils/personality_manager.py`
   - Update configuration

### Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_voice_cloning.py

# Run with coverage
pytest --cov=src tests/
```

## 🔍 Troubleshooting

### Common Issues

1. **Camera Not Working**:
   - Check camera permissions
   - Verify camera device ID in config
   - Test with `cv2.VideoCapture(0)`

2. **Voice Cloning Fails**:
   - Ensure audio quality (16kHz+, 3-30s duration)
   - Check YourTTS installation
   - Verify audio file format

3. **OpenAI API Errors**:
   - Verify API key is set
   - Check API quota and billing
   - Ensure internet connection

4. **Memory Issues**:
   - Reduce model sizes in config
   - Lower processing intervals
   - Close unused applications

### Performance Optimization

1. **GPU Acceleration**:
   ```yaml
   performance:
     gpu_acceleration: true
   ```

2. **Model Optimization**:
   - Use smaller YOLO models
   - Reduce processing frequency
   - Enable model caching

3. **Memory Management**:
   - Limit conversation history
   - Clear vision cache periodically
   - Use streaming for large audio

## 📝 API Reference

### Core Assistant Methods

```python
# Initialize
assistant = PersonalAssistant(config)

# Voice operations
assistant.clone_voice(audio_file, voice_name)
assistant.synthesize_speech(text, voice_name)
assistant.set_active_voice(voice_name)

# Vision operations
assistant.vision_processor.start_camera()
assistant.vision_processor.start_processing()
detections = assistant.object_detector.detect_objects(image_path)

# Conversation
response = assistant.process_text_input(user_message)
history = assistant.get_conversation_history()

# Personality
assistant.personality_manager.set_personality_preset("friendly")
assistant.personality_manager.update_personality(traits)
```

### Configuration Options

```python
# Load custom config
config = Config("path/to/custom_config.yaml")

# Update settings
config.update("voice.speech_recognition.language", "es")
config.save()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4 API
- **Coqui AI** for YourTTS
- **Ultralytics** for YOLOv8
- **OpenAI** for Whisper
- **Streamlit** for web interface


**Note**: This is a comprehensive AI assistant system. Ensure you have proper permissions and comply with relevant regulations when using voice cloning and camera features. 
