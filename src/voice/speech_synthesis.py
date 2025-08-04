"""
Speech Synthesis Module for Text-to-Speech with Voice Cloning
"""

import logging
import pyaudio
import wave
import io
import threading
import queue
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """Text-to-speech synthesis with voice cloning support."""
    
    def __init__(self, config):
        self.config = config
        self.voice_config = config.get_voice_config()
        self.audio_config = self.voice_config.get('audio', {})
        
        # Audio settings
        self.sample_rate = self.audio_config.get('sample_rate', 22050)
        self.channels = self.audio_config.get('channels', 1)
        self.format = pyaudio.paInt16
        
        # Audio playback
        self.audio = pyaudio.PyAudio()
        self.current_stream = None
        self.is_playing = False
        
        # Voice management
        self.active_voice = None
        self.voice_cloner = None  # Will be set by assistant
        
        # Audio queue for continuous playback
        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self.should_stop_playback = False
    
    def set_voice_cloner(self, voice_cloner):
        """Set the voice cloner instance."""
        self.voice_cloner = voice_cloner
    
    def set_voice(self, voice_name: str) -> bool:
        """Set the active voice for synthesis."""
        if self.voice_cloner:
            available_voices = self.voice_cloner.get_available_voices()
            if voice_name in available_voices:
                self.active_voice = voice_name
                logger.info(f"Active voice set to: {voice_name}")
                return True
            else:
                logger.warning(f"Voice not found: {voice_name}")
                return False
        else:
            logger.error("Voice cloner not set")
            return False
    
    def get_active_voice(self) -> Optional[str]:
        """Get the currently active voice."""
        return self.active_voice
    
    def synthesize(self, text: str, voice_name: Optional[str] = None) -> bytes:
        """Synthesize speech from text."""
        try:
            # Use specified voice or active voice
            target_voice = voice_name or self.active_voice
            
            if target_voice and self.voice_cloner:
                # Use cloned voice
                audio_data = self.voice_cloner.synthesize_with_voice(text, target_voice)
                if audio_data:
                    return audio_data
                else:
                    logger.warning(f"Failed to synthesize with voice {target_voice}, falling back to default")
            
            # Fall back to default TTS
            return self._synthesize_default(text)
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return b""
    
    def _synthesize_default(self, text: str) -> bytes:
        """Synthesize speech using default TTS (fallback)."""
        try:
            # For now, return empty audio as fallback
            # In a real implementation, you'd use a default TTS model
            logger.warning("Default TTS not implemented, returning empty audio")
            return b""
            
        except Exception as e:
            logger.error(f"Error in default synthesis: {e}")
            return b""
    
    def play_audio(self, audio_data: bytes):
        """Play audio data through speakers."""
        if not audio_data:
            return
        
        try:
            # Stop any currently playing audio
            self.stop_audio()
            
            # Start playback in separate thread
            self.playback_thread = threading.Thread(
                target=self._play_audio_thread,
                args=(audio_data,),
                daemon=True
            )
            self.playback_thread.start()
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def _play_audio_thread(self, audio_data: bytes):
        """Play audio in a separate thread."""
        try:
            self.is_playing = True
            
            # Open audio stream
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            
            # Play audio data
            stream.write(audio_data)
            
            # Clean up
            stream.stop_stream()
            stream.close()
            
            self.is_playing = False
            
        except Exception as e:
            logger.error(f"Error in audio playback thread: {e}")
            self.is_playing = False
    
    def stop_audio(self):
        """Stop currently playing audio."""
        self.is_playing = False
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
    
    def synthesize_and_play(self, text: str, voice_name: Optional[str] = None):
        """Synthesize speech and play it immediately."""
        audio_data = self.synthesize(text, voice_name)
        if audio_data:
            self.play_audio(audio_data)
    
    def save_audio(self, text: str, output_path: str, voice_name: Optional[str] = None) -> bool:
        """Synthesize speech and save to file."""
        try:
            audio_data = self.synthesize(text, voice_name)
            if not audio_data:
                return False
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Save as WAV file
            sf.write(output_path, audio_array, self.sample_rate)
            
            logger.info(f"Audio saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False
    
    def get_audio_duration(self, audio_data: bytes) -> float:
        """Get duration of audio data in seconds."""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            duration = len(audio_array) / self.sample_rate
            return duration
        except Exception as e:
            logger.error(f"Error calculating audio duration: {e}")
            return 0.0
    
    def adjust_speech_rate(self, audio_data: bytes, rate_factor: float) -> bytes:
        """Adjust speech rate of audio data."""
        try:
            if rate_factor == 1.0:
                return audio_data
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Resample audio
            from scipy import signal
            new_length = int(len(audio_array) / rate_factor)
            resampled = signal.resample(audio_array, new_length)
            
            # Convert back to bytes
            return resampled.astype(np.int16).tobytes()
            
        except Exception as e:
            logger.error(f"Error adjusting speech rate: {e}")
            return audio_data
    
    def adjust_pitch(self, audio_data: bytes, pitch_factor: float) -> bytes:
        """Adjust pitch of audio data."""
        try:
            if pitch_factor == 1.0:
                return audio_data
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Apply pitch shift
            import librosa
            audio_float = audio_array.astype(np.float32) / 32768.0
            shifted = librosa.effects.pitch_shift(audio_float, sr=self.sample_rate, n_steps=pitch_factor)
            
            # Convert back to bytes
            return (shifted * 32768).astype(np.int16).tobytes()
            
        except Exception as e:
            logger.error(f"Error adjusting pitch: {e}")
            return audio_data
    
    def get_available_voices(self) -> list:
        """Get list of available voices."""
        if self.voice_cloner:
            return self.voice_cloner.get_available_voices()
        return []
    
    def test_voice(self, voice_name: str) -> bool:
        """Test a voice with sample text."""
        try:
            test_text = "Hello, this is a test of the voice synthesis system."
            audio_data = self.synthesize(test_text, voice_name)
            
            if audio_data:
                logger.info(f"Voice test successful for: {voice_name}")
                return True
            else:
                logger.error(f"Voice test failed for: {voice_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing voice: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_audio()
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        logger.info("Speech synthesizer cleaned up") 