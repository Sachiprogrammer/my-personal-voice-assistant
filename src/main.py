#!/usr/bin/env python3
"""
My Personal Voice Assistant - Main Application
AI-Powered Personalized Assistant with Vision and Voice Cloning
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from core.assistant import PersonalAssistant
from core.config import Config
from ui.streamlit_app import create_streamlit_app
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


class VoiceAssistantApp:
    """Main application class for the AI-powered voice assistant."""
    
    def __init__(self):
        self.config = Config()
        self.assistant = None
        setup_logging()
        
    def initialize(self):
        """Initialize the assistant and all components."""
        try:
            logger.info("Initializing My Personal Voice Assistant...")
            
            # Initialize the personal assistant
            self.assistant = PersonalAssistant(self.config)
            
            logger.info("Assistant initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {e}")
            return False
    
    def run_cli(self):
        """Run the assistant in CLI mode."""
        if not self.initialize():
            return
        
        logger.info("Starting CLI mode...")
        print("🎤 My Personal Voice Assistant")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        try:
            while True:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input:
                    response = self.assistant.process_text_input(user_input)
                    print(f"Assistant: {response}")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            logger.error(f"Error in CLI mode: {e}")
    
    def run_streamlit(self):
        """Run the assistant with Streamlit UI."""
        if not self.initialize():
            return
        
        logger.info("Starting Streamlit UI...")
        create_streamlit_app(self.assistant)
    
    def run_voice_mode(self):
        """Run the assistant in voice-only mode."""
        if not self.initialize():
            return
        
        logger.info("Starting voice mode...")
        try:
            self.assistant.start_voice_conversation()
        except KeyboardInterrupt:
            logger.info("Voice mode stopped by user")
        except Exception as e:
            logger.error(f"Error in voice mode: {e}")


def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="My Personal Voice Assistant")
    parser.add_argument(
        "--mode", 
        choices=["cli", "web", "voice"], 
        default="web",
        help="Run mode: cli (command line), web (streamlit), voice (voice only)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/settings.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    app = VoiceAssistantApp()
    
    if args.mode == "cli":
        app.run_cli()
    elif args.mode == "web":
        app.run_streamlit()
    elif args.mode == "voice":
        app.run_voice_mode()


if __name__ == "__main__":
    main() 