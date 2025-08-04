"""
Core Personal Assistant with Voice Cloning and Vision Processing
"""

import asyncio
import logging
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import Config
from ..voice.voice_cloner import VoiceCloner
from ..voice.speech_recognition import SpeechRecognizer
from ..voice.speech_synthesis import SpeechSynthesizer
from ..vision.vision_processor import VisionProcessor
from ..vision.object_detection import ObjectDetector
from ..utils.conversation_manager import ConversationManager
from ..utils.personality_manager import PersonalityManager

logger = logging.getLogger(__name__)


class PersonalAssistant:
    """
    Main personal assistant class that integrates all components:
    - Voice cloning and synthesis
    - Speech recognition
    - Vision processing
    - Natural language understanding
    - Conversation management
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.voice_cloner = None
        self.speech_recognizer = None
        self.speech_synthesizer = None
        self.vision_processor = None
        self.object_detector = None
        self.conversation_manager = None
        self.personality_manager = None
        
        self.is_initialized = False
        self.is_listening = False
        self.voice_thread = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all assistant components."""
        try:
            logger.info("Initializing assistant components...")
            
            # Initialize voice components
            self.voice_cloner = VoiceCloner(self.config)
            self.speech_recognizer = SpeechRecognizer(self.config)
            self.speech_synthesizer = SpeechSynthesizer(self.config)
            
            # Initialize vision components
            self.vision_processor = VisionProcessor(self.config)
            self.object_detector = ObjectDetector(self.config)
            
            # Initialize conversation and personality managers
            self.conversation_manager = ConversationManager(self.config)
            self.personality_manager = PersonalityManager(self.config)
            
            # Set up component relationships
            self._setup_component_relationships()
            
            self.is_initialized = True
            logger.info("All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_component_relationships(self):
        """Set up relationships between components."""
        try:
            # Connect voice cloner to speech synthesizer
            self.speech_synthesizer.set_voice_cloner(self.voice_cloner)
            
            # Connect object detector to vision processor
            self.vision_processor.set_object_detector(self.object_detector)
            
            # Connect vision processor to conversation manager
            self.conversation_manager.set_vision_processor(self.vision_processor)
            
            logger.info("Component relationships established")
            
        except Exception as e:
            logger.error(f"Failed to setup component relationships: {e}")
            raise
    
    def process_text_input(self, text: str) -> str:
        """Process text input and return response."""
        if not self.is_initialized:
            return "Assistant is not initialized. Please try again."
        
        try:
            # Add to conversation history
            self.conversation_manager.add_user_message(text)
            
            # Get personality context
            personality_context = self.personality_manager.get_context()
            
            # Generate response using conversation manager
            response = self.conversation_manager.generate_response(
                text, personality_context
            )
            
            # Add assistant response to history
            self.conversation_manager.add_assistant_message(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    def process_voice_input(self, audio_data: bytes) -> str:
        """Process voice input and return text response."""
        if not self.is_initialized:
            return "Assistant is not initialized. Please try again."
        
        try:
            # Convert speech to text
            text = self.speech_recognizer.recognize(audio_data)
            
            if not text:
                return "I couldn't understand what you said. Please try again."
            
            # Process the text input
            response = self.process_text_input(text)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            return "I'm sorry, I encountered an error processing your voice input."
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image and return analysis results."""
        if not self.is_initialized:
            return {"error": "Assistant is not initialized"}
        
        try:
            # Detect objects in the image
            objects = self.object_detector.detect_objects(image_path)
            
            # Process image with vision processor
            analysis = self.vision_processor.analyze_image(image_path)
            
            # Combine results
            results = {
                "objects": objects,
                "analysis": analysis,
                "summary": self.vision_processor.generate_summary(analysis)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": f"Failed to process image: {str(e)}"}
    
    def clone_voice(self, audio_file: str, voice_name: str) -> bool:
        """Clone a voice from audio file."""
        if not self.is_initialized:
            return False
        
        try:
            success = self.voice_cloner.clone_voice(audio_file, voice_name)
            if success:
                logger.info(f"Successfully cloned voice: {voice_name}")
            return success
            
        except Exception as e:
            logger.error(f"Error cloning voice: {e}")
            return False
    
    def synthesize_speech(self, text: str, voice_name: Optional[str] = None) -> bytes:
        """Synthesize speech from text using cloned voice if available."""
        if not self.is_initialized:
            return b""
        
        try:
            audio_data = self.speech_synthesizer.synthesize(text, voice_name)
            return audio_data
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return b""
    
    def start_voice_conversation(self):
        """Start continuous voice conversation."""
        if not self.is_initialized:
            logger.error("Assistant not initialized")
            return
        
        self.is_listening = True
        logger.info("Starting voice conversation...")
        
        try:
            while self.is_listening:
                # Listen for voice input
                audio_data = self.speech_recognizer.listen()
                
                if audio_data:
                    # Process voice input
                    response = self.process_voice_input(audio_data)
                    
                    # Synthesize and play response
                    audio_response = self.synthesize_speech(response)
                    if audio_response:
                        self.speech_synthesizer.play_audio(audio_response)
                
        except KeyboardInterrupt:
            logger.info("Voice conversation stopped by user")
        except Exception as e:
            logger.error(f"Error in voice conversation: {e}")
        finally:
            self.is_listening = False
    
    def stop_voice_conversation(self):
        """Stop voice conversation."""
        self.is_listening = False
        logger.info("Voice conversation stopped")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_manager.get_history()
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_manager.clear_history()
        logger.info("Conversation history cleared")
    
    def update_personality(self, personality_traits: Dict[str, Any]):
        """Update assistant personality."""
        self.personality_manager.update_personality(personality_traits)
        logger.info("Personality updated")
    
    def get_available_voices(self) -> List[str]:
        """Get list of available cloned voices."""
        return self.voice_cloner.get_available_voices()
    
    def set_active_voice(self, voice_name: str) -> bool:
        """Set the active voice for synthesis."""
        return self.speech_synthesizer.set_voice(voice_name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get assistant status and capabilities."""
        return {
            "initialized": self.is_initialized,
            "listening": self.is_listening,
            "available_voices": self.get_available_voices(),
            "conversation_length": len(self.get_conversation_history()),
            "personality": self.personality_manager.get_personality()
        } 