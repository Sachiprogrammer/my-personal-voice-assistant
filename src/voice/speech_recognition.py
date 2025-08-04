"""
Speech Recognition Module for Real-time Voice Input
"""

import logging
import numpy as np
import pyaudio
import wave
import threading
import queue
from typing import Optional, Callable
import whisper
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """Real-time speech recognition using Whisper."""
    
    def __init__(self, config):
        self.config = config
        self.voice_config = config.get_voice_config()
        self.audio_config = self.voice_config.get('audio', {})
        self.sr_config = self.voice_config.get('speech_recognition', {})
        
        # Audio settings
        self.sample_rate = self.sr_config.get('sample_rate', 16000)
        self.chunk_size = self.sr_config.get('chunk_size', 1024)
        self.silence_threshold = self.sr_config.get('silence_threshold', 0.01)
        self.silence_duration = self.sr_config.get('silence_duration', 1.0)
        self.channels = self.audio_config.get('channels', 1)
        self.format = pyaudio.paInt16
        
        # Initialize Whisper model
        self.model = None
        self._load_whisper_model()
        
        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = False
        
        # Audio buffer
        self.audio_buffer = []
        self.silence_frames = 0
        self.silence_threshold_frames = int(self.silence_duration * self.sample_rate / self.chunk_size)
        
        # Callback for real-time processing
        self.on_speech_detected: Optional[Callable] = None
    
    def _load_whisper_model(self):
        """Load Whisper model for speech recognition."""
        try:
            model_name = self.sr_config.get('model', 'whisper')
            if model_name == 'whisper':
                model_name = 'base'  # Use base model for faster processing
            
            logger.info(f"Loading Whisper model: {model_name}")
            self.model = whisper.load_model(model_name)
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def start_listening(self, callback: Optional[Callable] = None):
        """Start listening for speech input."""
        if self.is_listening:
            logger.warning("Already listening for speech")
            return
        
        self.on_speech_detected = callback
        self.is_listening = True
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            logger.info("Started listening for speech input")
            
        except Exception as e:
            logger.error(f"Failed to start listening: {e}")
            self.is_listening = False
            raise
    
    def stop_listening(self):
        """Stop listening for speech input."""
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        logger.info("Stopped listening for speech input")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream processing."""
        if not self.is_listening:
            return (None, pyaudio.paComplete)
        
        # Convert audio data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Check for silence
        audio_level = np.sqrt(np.mean(audio_data**2))
        
        if audio_level < self.silence_threshold:
            self.silence_frames += 1
        else:
            self.silence_frames = 0
            self.audio_buffer.append(audio_data)
        
        # Process speech when silence is detected
        if (self.silence_frames >= self.silence_threshold_frames and 
            len(self.audio_buffer) > 0):
            
            # Combine audio chunks
            full_audio = np.concatenate(self.audio_buffer)
            
            # Process in separate thread to avoid blocking
            threading.Thread(
                target=self._process_audio_chunk,
                args=(full_audio,),
                daemon=True
            ).start()
            
            # Clear buffer
            self.audio_buffer = []
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_chunk(self, audio_data: np.ndarray):
        """Process audio chunk and convert to text."""
        try:
            # Normalize audio
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_data,
                language=self.sr_config.get('language', 'en'),
                fp16=False
            )
            
            text = result['text'].strip()
            
            if text and self.on_speech_detected:
                self.on_speech_detected(text)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def listen(self) -> Optional[str]:
        """Listen for a single speech input and return transcribed text."""
        if self.is_listening:
            logger.warning("Already listening, cannot start new listen session")
            return None
        
        result_queue = queue.Queue()
        
        def on_speech(text):
            result_queue.put(text)
        
        try:
            self.start_listening(on_speech)
            
            # Wait for speech input
            text = result_queue.get(timeout=30)  # 30 second timeout
            return text
            
        except queue.Empty:
            logger.info("No speech detected within timeout")
            return None
        finally:
            self.stop_listening()
    
    def recognize(self, audio_data: bytes) -> str:
        """Recognize speech from audio data."""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_array,
                language=self.sr_config.get('language', 'en'),
                fp16=False
            )
            
            return result['text'].strip()
            
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            return ""
    
    def recognize_file(self, audio_file: str) -> str:
        """Recognize speech from audio file."""
        try:
            result = self.model.transcribe(
                audio_file,
                language=self.sr_config.get('language', 'en'),
                fp16=False
            )
            
            return result['text'].strip()
            
        except Exception as e:
            logger.error(f"Error recognizing speech from file: {e}")
            return ""
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_listening()
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        logger.info("Speech recognizer cleaned up") 