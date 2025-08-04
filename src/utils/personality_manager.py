"""
Personality Manager for AI Assistant
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PersonalityManager:
    """Manages the assistant's personality and behavior traits."""
    
    def __init__(self, config):
        self.config = config
        self.conversation_config = config.get_conversation_config()
        self.personality_config = self.conversation_config.get('conversation', {}).get('personality', {})
        
        # Default personality
        self.personality = {
            'name': self.personality_config.get('name', 'Alex'),
            'traits': self.personality_config.get('traits', {
                'friendly': 0.8,
                'helpful': 0.9,
                'professional': 0.7,
                'humorous': 0.3
            }),
            'voice_style': 'conversational',
            'response_style': 'detailed',
            'interests': ['technology', 'helping people', 'learning'],
            'conversation_topics': ['general knowledge', 'assistance', 'casual chat']
        }
        
        # Personality presets
        self.personality_presets = {
            'professional': {
                'name': 'Alex',
                'traits': {
                    'friendly': 0.6,
                    'helpful': 0.9,
                    'professional': 0.9,
                    'humorous': 0.2
                },
                'voice_style': 'professional',
                'response_style': 'concise',
                'interests': ['efficiency', 'accuracy', 'professional development'],
                'conversation_topics': ['work', 'productivity', 'professional advice']
            },
            'friendly': {
                'name': 'Alex',
                'traits': {
                    'friendly': 0.9,
                    'helpful': 0.8,
                    'professional': 0.5,
                    'humorous': 0.7
                },
                'voice_style': 'warm',
                'response_style': 'conversational',
                'interests': ['people', 'relationships', 'fun'],
                'conversation_topics': ['casual chat', 'personal interests', 'entertainment']
            },
            'casual': {
                'name': 'Alex',
                'traits': {
                    'friendly': 0.8,
                    'helpful': 0.7,
                    'professional': 0.4,
                    'humorous': 0.8
                },
                'voice_style': 'casual',
                'response_style': 'relaxed',
                'interests': ['entertainment', 'casual conversation', 'fun'],
                'conversation_topics': ['casual topics', 'entertainment', 'personal stories']
            }
        }
    
    def get_personality(self) -> Dict[str, Any]:
        """Get current personality settings."""
        return self.personality.copy()
    
    def update_personality(self, personality_traits: Dict[str, Any]):
        """Update personality traits."""
        try:
            # Update traits
            if 'traits' in personality_traits:
                self.personality['traits'].update(personality_traits['traits'])
            
            # Update other personality aspects
            for key, value in personality_traits.items():
                if key != 'traits':
                    self.personality[key] = value
            
            logger.info("Personality updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating personality: {e}")
    
    def set_personality_preset(self, preset_name: str) -> bool:
        """Set personality to a predefined preset."""
        try:
            if preset_name not in self.personality_presets:
                logger.warning(f"Personality preset '{preset_name}' not found")
                return False
            
            self.personality = self.personality_presets[preset_name].copy()
            logger.info(f"Personality set to '{preset_name}' preset")
            return True
            
        except Exception as e:
            logger.error(f"Error setting personality preset: {e}")
            return False
    
    def get_available_presets(self) -> List[str]:
        """Get list of available personality presets."""
        return list(self.personality_presets.keys())
    
    def get_context(self) -> str:
        """Get personality context for conversation."""
        try:
            traits = self.personality['traits']
            name = self.personality['name']
            
            # Build personality description
            trait_descriptions = []
            
            if traits.get('friendly', 0) > 0.7:
                trait_descriptions.append("very friendly and approachable")
            elif traits.get('friendly', 0) > 0.5:
                trait_descriptions.append("friendly")
            
            if traits.get('helpful', 0) > 0.8:
                trait_descriptions.append("extremely helpful")
            elif traits.get('helpful', 0) > 0.6:
                trait_descriptions.append("helpful")
            
            if traits.get('professional', 0) > 0.8:
                trait_descriptions.append("highly professional")
            elif traits.get('professional', 0) > 0.6:
                trait_descriptions.append("professional")
            
            if traits.get('humorous', 0) > 0.7:
                trait_descriptions.append("with a good sense of humor")
            elif traits.get('humorous', 0) > 0.5:
                trait_descriptions.append("slightly humorous")
            
            personality_desc = f"{name} is {' and '.join(trait_descriptions)}."
            
            # Add interests
            if self.personality.get('interests'):
                interests = ', '.join(self.personality['interests'])
                personality_desc += f" I'm interested in {interests}."
            
            # Add conversation style
            response_style = self.personality.get('response_style', 'conversational')
            personality_desc += f" I prefer to give {response_style} responses."
            
            return personality_desc
            
        except Exception as e:
            logger.error(f"Error generating personality context: {e}")
            return "I'm a helpful AI assistant."
    
    def get_response_style(self) -> str:
        """Get the current response style."""
        return self.personality.get('response_style', 'conversational')
    
    def get_voice_style(self) -> str:
        """Get the current voice style."""
        return self.personality.get('voice_style', 'conversational')
    
    def get_name(self) -> str:
        """Get the assistant's name."""
        return self.personality.get('name', 'Alex')
    
    def should_use_humor(self) -> bool:
        """Determine if humor should be used based on personality."""
        return self.personality['traits'].get('humorous', 0) > 0.5
    
    def should_be_formal(self) -> bool:
        """Determine if formal language should be used."""
        return self.personality['traits'].get('professional', 0) > 0.7
    
    def should_be_friendly(self) -> bool:
        """Determine if friendly language should be used."""
        return self.personality['traits'].get('friendly', 0) > 0.6
    
    def get_conversation_topics(self) -> List[str]:
        """Get preferred conversation topics."""
        return self.personality.get('conversation_topics', [])
    
    def get_interests(self) -> List[str]:
        """Get assistant's interests."""
        return self.personality.get('interests', [])
    
    def adjust_response_tone(self, response: str) -> str:
        """Adjust response tone based on personality."""
        try:
            # This is a simple implementation - in a real system, you'd use more sophisticated NLP
            if self.should_be_formal():
                # Make response more formal
                response = response.replace("I'm", "I am")
                response = response.replace("don't", "do not")
                response = response.replace("can't", "cannot")
            
            if self.should_be_friendly():
                # Add friendly elements
                if not response.startswith(("Hello", "Hi", "Hey")):
                    response = f"Hey there! {response}"
            
            if self.should_use_humor() and len(response) > 50:
                # Add a light touch (very simple implementation)
                if "sorry" in response.lower():
                    response += " 😊"
            
            return response
            
        except Exception as e:
            logger.error(f"Error adjusting response tone: {e}")
            return response
    
    def save_personality(self, file_path: str) -> bool:
        """Save personality settings to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.personality, f, indent=2)
            
            logger.info(f"Personality saved to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving personality: {e}")
            return False
    
    def load_personality(self, file_path: str) -> bool:
        """Load personality settings from file."""
        try:
            with open(file_path, 'r') as f:
                loaded_personality = json.load(f)
            
            # Validate personality structure
            required_keys = ['name', 'traits']
            for key in required_keys:
                if key not in loaded_personality:
                    raise ValueError(f"Missing required personality key: {key}")
            
            self.personality = loaded_personality
            logger.info(f"Personality loaded from: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading personality: {e}")
            return False
    
    def create_custom_preset(self, preset_name: str, personality_data: Dict[str, Any]) -> bool:
        """Create a custom personality preset."""
        try:
            # Validate personality data
            if 'name' not in personality_data or 'traits' not in personality_data:
                raise ValueError("Personality data must include 'name' and 'traits'")
            
            self.personality_presets[preset_name] = personality_data.copy()
            logger.info(f"Custom personality preset '{preset_name}' created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating custom preset: {e}")
            return False
    
    def delete_preset(self, preset_name: str) -> bool:
        """Delete a custom personality preset."""
        try:
            if preset_name in ['professional', 'friendly', 'casual']:
                logger.warning("Cannot delete built-in presets")
                return False
            
            if preset_name in self.personality_presets:
                del self.personality_presets[preset_name]
                logger.info(f"Personality preset '{preset_name}' deleted")
                return True
            else:
                logger.warning(f"Preset '{preset_name}' not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting preset: {e}")
            return False
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """Get a summary of the current personality."""
        try:
            traits = self.personality['traits']
            
            # Calculate dominant traits
            dominant_traits = []
            for trait, value in traits.items():
                if value > 0.7:
                    dominant_traits.append(trait)
                elif value < 0.3:
                    dominant_traits.append(f"not {trait}")
            
            return {
                'name': self.personality['name'],
                'dominant_traits': dominant_traits,
                'response_style': self.personality.get('response_style', 'conversational'),
                'voice_style': self.personality.get('voice_style', 'conversational'),
                'interests': self.personality.get('interests', []),
                'conversation_topics': self.personality.get('conversation_topics', [])
            }
            
        except Exception as e:
            logger.error(f"Error generating personality summary: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up resources."""
        # Clear any temporary data
        logger.info("Personality manager cleaned up") 