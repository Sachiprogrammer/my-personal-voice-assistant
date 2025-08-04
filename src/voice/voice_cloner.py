"""
Voice Cloning Module for Personalized Voice Synthesis
"""

import logging
import os
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import librosa
import soundfile as sf
from TTS.api import TTS

logger = logging.getLogger(__name__)


class VoiceCloner:
    """Voice cloning and management using YourTTS."""
    
    def __init__(self, config):
        self.config = config
        self.voice_config = config.get_voice_config()
        self.vc_config = self.voice_config.get('speech_synthesis', {}).get('voice_cloning', {})
        
        # Paths
        self.voice_samples_dir = Path(self.vc_config.get('voice_samples_dir', 'assets/audio/voice_samples'))
        self.cloned_voices_dir = Path(self.vc_config.get('cloned_voices_dir', 'assets/models/voice_clones'))
        
        # Create directories if they don't exist
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        self.cloned_voices_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TTS model
        self.tts_model = None
        self._load_tts_model()
        
        # Voice registry
        self.voice_registry = self._load_voice_registry()
    
    def _load_tts_model(self):
        """Load YourTTS model for voice cloning."""
        try:
            model_name = self.vc_config.get('model', 'your-tts')
            
            logger.info(f"Loading TTS model: {model_name}")
            
            # Initialize YourTTS model
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts")
            
            logger.info("TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise
    
    def _load_voice_registry(self) -> Dict[str, Dict]:
        """Load voice registry from file."""
        registry_file = self.cloned_voices_dir / "voice_registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load voice registry: {e}")
        
        return {}
    
    def _save_voice_registry(self):
        """Save voice registry to file."""
        registry_file = self.cloned_voices_dir / "voice_registry.json"
        
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.voice_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save voice registry: {e}")
    
    def clone_voice(self, audio_file: str, voice_name: str) -> bool:
        """Clone a voice from audio file."""
        try:
            audio_path = Path(audio_file)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_file}")
                return False
            
            # Validate audio file
            if not self._validate_audio_file(audio_path):
                logger.error(f"Invalid audio file: {audio_file}")
                return False
            
            # Create voice directory
            voice_dir = self.cloned_voices_dir / voice_name
            voice_dir.mkdir(exist_ok=True)
            
            # Copy audio file to voice samples
            sample_path = self.voice_samples_dir / f"{voice_name}_sample.wav"
            self._prepare_audio_sample(audio_path, sample_path)
            
            # Register voice
            self.voice_registry[voice_name] = {
                "sample_path": str(sample_path),
                "created_at": str(Path().cwd()),
                "status": "cloned"
            }
            
            self._save_voice_registry()
            
            logger.info(f"Successfully cloned voice: {voice_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clone voice: {e}")
            return False
    
    def _validate_audio_file(self, audio_path: Path) -> bool:
        """Validate audio file format and quality."""
        try:
            # Check file format
            if audio_path.suffix.lower() not in ['.wav', '.mp3', '.flac', '.m4a']:
                return False
            
            # Load audio to check quality
            audio, sr = librosa.load(str(audio_path), sr=None)
            
            # Check duration (minimum 3 seconds, maximum 30 seconds)
            duration = len(audio) / sr
            if duration < 3.0 or duration > 30.0:
                logger.warning(f"Audio duration ({duration:.1f}s) is outside recommended range (3-30s)")
            
            # Check sample rate
            if sr < 16000:
                logger.warning(f"Low sample rate ({sr}Hz), may affect quality")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio file: {e}")
            return False
    
    def _prepare_audio_sample(self, input_path: Path, output_path: Path):
        """Prepare audio sample for voice cloning."""
        try:
            # Load audio
            audio, sr = librosa.load(str(input_path), sr=22050)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Save as WAV
            sf.write(str(output_path), audio, sr)
            
        except Exception as e:
            logger.error(f"Error preparing audio sample: {e}")
            raise
    
    def get_available_voices(self) -> List[str]:
        """Get list of available cloned voices."""
        return list(self.voice_registry.keys())
    
    def get_voice_info(self, voice_name: str) -> Optional[Dict]:
        """Get information about a specific voice."""
        return self.voice_registry.get(voice_name)
    
    def delete_voice(self, voice_name: str) -> bool:
        """Delete a cloned voice."""
        try:
            if voice_name not in self.voice_registry:
                logger.warning(f"Voice not found: {voice_name}")
                return False
            
            # Remove voice files
            voice_dir = self.cloned_voices_dir / voice_name
            if voice_dir.exists():
                import shutil
                shutil.rmtree(voice_dir)
            
            # Remove sample file
            sample_path = Path(self.voice_registry[voice_name]['sample_path'])
            if sample_path.exists():
                sample_path.unlink()
            
            # Remove from registry
            del self.voice_registry[voice_name]
            self._save_voice_registry()
            
            logger.info(f"Deleted voice: {voice_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete voice: {e}")
            return False
    
    def synthesize_with_voice(self, text: str, voice_name: str) -> Optional[bytes]:
        """Synthesize speech with cloned voice."""
        try:
            if voice_name not in self.voice_registry:
                logger.error(f"Voice not found: {voice_name}")
                return None
            
            voice_info = self.voice_registry[voice_name]
            sample_path = voice_info['sample_path']
            
            # Generate speech with cloned voice
            output_path = self.cloned_voices_dir / f"{voice_name}_temp.wav"
            
            self.tts_model.tts_to_file(
                text=text,
                speaker_wav=sample_path,
                file_path=str(output_path)
            )
            
            # Read generated audio
            with open(output_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temp file
            output_path.unlink()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to synthesize with voice {voice_name}: {e}")
            return None
    
    def test_voice_clone(self, voice_name: str) -> bool:
        """Test a cloned voice with a sample text."""
        try:
            test_text = "Hello, this is a test of the voice cloning system."
            audio_data = self.synthesize_with_voice(test_text, voice_name)
            
            if audio_data:
                logger.info(f"Voice clone test successful for: {voice_name}")
                return True
            else:
                logger.error(f"Voice clone test failed for: {voice_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing voice clone: {e}")
            return False
    
    def get_voice_quality_metrics(self, voice_name: str) -> Dict:
        """Get quality metrics for a cloned voice."""
        try:
            voice_info = self.voice_registry.get(voice_name)
            if not voice_info:
                return {"error": "Voice not found"}
            
            sample_path = voice_info['sample_path']
            
            # Load sample audio
            audio, sr = librosa.load(sample_path, sr=None)
            
            # Calculate metrics
            duration = len(audio) / sr
            rms_energy = np.sqrt(np.mean(audio**2))
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "rms_energy": float(rms_energy),
                "spectral_centroid": float(spectral_centroid),
                "status": voice_info.get('status', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error getting voice quality metrics: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up resources."""
        if self.tts_model:
            del self.tts_model
            self.tts_model = None
        
        logger.info("Voice cloner cleaned up") 