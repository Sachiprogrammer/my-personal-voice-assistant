"""
Conversation Manager for AI Assistant with Vision Integration
"""

import logging
import openai
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversations with vision-aware AI assistant."""
    
    def __init__(self, config):
        self.config = config
        self.conversation_config = config.get_conversation_config()
        self.llm_config = self.conversation_config.get('llm', {})
        self.conv_config = self.conversation_config.get('conversation', {})
        
        # OpenAI settings
        self.api_key = config.get_api_keys().get('openai', {}).get('api_key')
        if not self.api_key:
            logger.error("OpenAI API key not found")
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.api_key
        
        # Model settings
        self.model = self.llm_config.get('model', 'gpt-4')
        self.temperature = self.llm_config.get('temperature', 0.7)
        self.max_tokens = self.llm_config.get('max_tokens', 500)
        self.system_prompt = self.llm_config.get('system_prompt', '')
        
        # Conversation history
        self.max_history = self.conv_config.get('max_history', 50)
        self.context_window = self.conv_config.get('context_window', 10)
        self.conversation_history = []
        
        # Vision integration
        self.vision_processor = None  # Will be set by assistant
        self.last_vision_update = None
    
    def set_vision_processor(self, vision_processor):
        """Set the vision processor for spatial awareness."""
        self.vision_processor = vision_processor
    
    def add_user_message(self, message: str):
        """Add a user message to conversation history."""
        self.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Maintain history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to conversation history."""
        self.conversation_history.append({
            'role': 'assistant',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Maintain history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_vision_context(self) -> str:
        """Get current vision context for the conversation."""
        if not self.vision_processor:
            return ""
        
        try:
            vision_state = self.vision_processor.get_current_vision_state()
            detections = vision_state.get('detections', [])
            relationships = vision_state.get('relationships', [])
            
            if not detections:
                return "I don't see any objects in the environment right now."
            
            # Build vision context
            context_parts = []
            
            # Object summary
            object_counts = {}
            for detection in detections:
                obj_class = detection['class']
                object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
            
            obj_list = []
            for obj_name, count in object_counts.items():
                if count == 1:
                    obj_list.append(f"a {obj_name}")
                else:
                    obj_list.append(f"{count} {obj_name}s")
            
            context_parts.append(f"I can see {', '.join(obj_list)} in the environment.")
            
            # Spatial relationships
            if relationships:
                rel_list = []
                for rel in relationships[:3]:  # Limit to 3 relationships
                    rel_desc = f"the {rel['object1']} is {rel['relationship']} the {rel['object2']}"
                    rel_list.append(rel_desc)
                
                if rel_list:
                    context_parts.append(f"Spatial relationships: {', '.join(rel_list)}.")
            
            return " ".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting vision context: {e}")
            return "I'm having trouble seeing the environment right now."
    
    def generate_response(self, user_message: str, personality_context: str = "") -> str:
        """Generate a response using OpenAI with vision integration."""
        try:
            # Get vision context
            vision_context = self.get_vision_context()
            
            # Build system prompt with context
            full_system_prompt = self.system_prompt
            if personality_context:
                full_system_prompt += f"\n\nPersonality: {personality_context}"
            
            if vision_context:
                full_system_prompt += f"\n\nCurrent Vision Context: {vision_context}"
            
            # Prepare messages for OpenAI
            messages = [{"role": "system", "content": full_system_prompt}]
            
            # Add conversation history (limited to context window)
            recent_history = self.conversation_history[-self.context_window:]
            for msg in recent_history:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            assistant_response = response.choices[0].message.content.strip()
            
            # Add to history
            self.add_assistant_message(assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now."
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        try:
            if not self.conversation_history:
                return {"message": "No conversation history"}
            
            # Count messages by role
            user_messages = len([msg for msg in self.conversation_history if msg['role'] == 'user'])
            assistant_messages = len([msg for msg in self.conversation_history if msg['role'] == 'assistant'])
            
            # Get recent topics (simple keyword extraction)
            recent_messages = self.conversation_history[-10:]  # Last 10 messages
            all_text = " ".join([msg['content'] for msg in recent_messages])
            
            # Simple keyword extraction (in a real implementation, you'd use NLP)
            keywords = []
            common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            words = all_text.lower().split()
            word_counts = {}
            
            for word in words:
                if word not in common_words and len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Get top keywords
            keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'total_messages': len(self.conversation_history),
                'user_messages': user_messages,
                'assistant_messages': assistant_messages,
                'recent_keywords': [word for word, count in keywords],
                'last_message_time': self.conversation_history[-1]['timestamp'] if self.conversation_history else None
            }
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return {"error": str(e)}
    
    def export_conversation(self, file_path: str) -> bool:
        """Export conversation history to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            
            logger.info(f"Conversation exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting conversation: {e}")
            return False
    
    def import_conversation(self, file_path: str) -> bool:
        """Import conversation history from file."""
        try:
            with open(file_path, 'r') as f:
                imported_history = json.load(f)
            
            # Validate format
            for msg in imported_history:
                if 'role' not in msg or 'content' not in msg:
                    raise ValueError("Invalid conversation format")
            
            self.conversation_history = imported_history
            logger.info(f"Conversation imported from: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing conversation: {e}")
            return False
    
    def get_response_with_vision(self, user_message: str, personality_context: str = "") -> str:
        """Generate response with explicit vision integration."""
        try:
            # Check if the message is about the environment
            environment_keywords = [
                'see', 'look', 'what', 'where', 'object', 'thing', 'room', 'environment',
                'table', 'chair', 'book', 'laptop', 'phone', 'cup', 'bottle', 'lamp',
                'tv', 'remote', 'left', 'right', 'above', 'below', 'near', 'next to'
            ]
            
            message_lower = user_message.lower()
            is_environment_query = any(keyword in message_lower for keyword in environment_keywords)
            
            if is_environment_query and self.vision_processor:
                # Force vision update
                vision_context = self.get_vision_context()
                logger.info(f"Environment query detected, vision context: {vision_context}")
            
            return self.generate_response(user_message, personality_context)
            
        except Exception as e:
            logger.error(f"Error generating vision-aware response: {e}")
            return "I'm sorry, I'm having trouble understanding the environment right now."
    
    def cleanup(self):
        """Clean up resources."""
        # Clear history to free memory
        self.conversation_history.clear()
        logger.info("Conversation manager cleaned up") 