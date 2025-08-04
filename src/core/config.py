"""
Configuration management for the Personal Voice Assistant
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for the Personal Voice Assistant."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/settings.yaml"
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Replace environment variables
            config = self._replace_env_vars(config)
            
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def _replace_env_vars(self, config: Any) -> Any:
        """Recursively replace environment variable placeholders."""
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(v) for v in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, "")
        else:
            return config
    
    def _validate_config(self):
        """Validate required configuration sections."""
        required_sections = ['voice', 'vision', 'conversation', 'app', 'paths']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_voice_config(self) -> Dict[str, Any]:
        """Get voice-related configuration."""
        return self.config.get('voice', {})
    
    def get_vision_config(self) -> Dict[str, Any]:
        """Get vision-related configuration."""
        return self.config.get('vision', {})
    
    def get_conversation_config(self) -> Dict[str, Any]:
        """Get conversation-related configuration."""
        return self.config.get('conversation', {})
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.config.get('app', {})
    
    def get_paths(self) -> Dict[str, str]:
        """Get file paths configuration."""
        return self.config.get('paths', {})
    
    def get_api_keys(self) -> Dict[str, Any]:
        """Get API keys configuration."""
        return self.config.get('api', {})
    
    def update(self, key: str, value: Any):
        """Update configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {e}")
    
    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
        self._validate_config() 