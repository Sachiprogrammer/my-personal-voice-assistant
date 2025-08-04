"""
Streamlit Web Interface for Personal Voice Assistant
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_streamlit_app(assistant):
    """Create the Streamlit web interface."""
    
    st.set_page_config(
        page_title="My Personal Voice Assistant",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 My Personal Voice Assistant")
    st.markdown("AI-powered assistant with voice cloning, vision, and spatial awareness")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # Voice controls
        st.subheader("Voice Settings")
        available_voices = assistant.get_available_voices()
        if available_voices:
            selected_voice = st.selectbox("Select Voice", available_voices)
            if st.button("Set Active Voice"):
                assistant.set_active_voice(selected_voice)
                st.success(f"Active voice set to: {selected_voice}")
        else:
            st.info("No cloned voices available")
        
        # Voice cloning
        st.subheader("Voice Cloning")
        uploaded_audio = st.file_uploader("Upload audio sample for voice cloning", type=['wav', 'mp3', 'flac'])
        if uploaded_audio and st.button("Clone Voice"):
            with st.spinner("Cloning voice..."):
                # Save uploaded file temporarily
                temp_path = f"temp_voice_sample.{uploaded_audio.name.split('.')[-1]}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_audio.getvalue())
                
                voice_name = st.text_input("Enter voice name", value="cloned_voice")
                if st.button("Confirm Clone"):
                    success = assistant.clone_voice(temp_path, voice_name)
                    if success:
                        st.success(f"Voice '{voice_name}' cloned successfully!")
                    else:
                        st.error("Failed to clone voice")
        
        # Vision controls
        st.subheader("Vision Settings")
        if st.button("Start Camera"):
            if assistant.vision_processor.start_camera():
                st.success("Camera started")
            else:
                st.error("Failed to start camera")
        
        if st.button("Start Vision Processing"):
            if assistant.vision_processor.start_processing():
                st.success("Vision processing started")
            else:
                st.error("Failed to start vision processing")
        
        if st.button("Stop Vision"):
            assistant.vision_processor.stop_processing()
            assistant.vision_processor.stop_camera()
            st.info("Vision processing stopped")
        
        # Personality controls
        st.subheader("Personality")
        personality_presets = assistant.personality_manager.get_available_presets()
        selected_preset = st.selectbox("Personality Preset", personality_presets)
        if st.button("Set Personality"):
            assistant.personality_manager.set_personality_preset(selected_preset)
            st.success(f"Personality set to: {selected_preset}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Conversation")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = assistant.process_text_input(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("Vision Feed")
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        
        # Vision status
        if assistant.vision_processor:
            vision_state = assistant.vision_processor.get_current_vision_state()
            
            if vision_state.get('camera_active'):
                st.success("Camera Active")
                
                # Display current detections
                detections = vision_state.get('detections', [])
                if detections:
                    st.subheader("Detected Objects")
                    for detection in detections[:5]:  # Show top 5
                        st.write(f"• {detection['class']} (confidence: {detection['confidence']:.2f})")
                
                # Display relationships
                relationships = vision_state.get('relationships', [])
                if relationships:
                    st.subheader("Spatial Relationships")
                    for rel in relationships[:3]:  # Show top 3
                        st.write(f"• {rel['object1']} is {rel['relationship']} {rel['object2']}")
            else:
                st.info("Camera not active")
        
        # Vision controls
        if st.button("Capture Frame"):
            if assistant.vision_processor:
                frame = assistant.vision_processor.capture_frame()
                if frame is not None:
                    # Convert to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    st.image(pil_image, caption="Captured Frame", use_column_width=True)
                else:
                    st.warning("No frame available")
    
    # Status section
    st.header("System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Voice")
        st.write(f"Initialized: {assistant.is_initialized}")
        if assistant.speech_synthesizer:
            active_voice = assistant.speech_synthesizer.get_active_voice()
            st.write(f"Active Voice: {active_voice or 'None'}")
    
    with col2:
        st.subheader("Vision")
        if assistant.vision_processor:
            vision_state = assistant.vision_processor.get_current_vision_state()
            st.write(f"Camera Active: {vision_state.get('camera_active', False)}")
            st.write(f"Processing Active: {vision_state.get('processing_active', False)}")
    
    with col3:
        st.subheader("Conversation")
        if assistant.conversation_manager:
            history = assistant.conversation_manager.get_history()
            st.write(f"Messages: {len(history)}")
            
            # Conversation summary
            summary = assistant.conversation_manager.get_conversation_summary()
            if 'total_messages' in summary:
                st.write(f"Total Messages: {summary['total_messages']}")
    
    # Voice interaction section
    st.header("Voice Interaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Voice Conversation"):
            st.info("Voice conversation started. Speak into your microphone.")
            # Note: In a real implementation, you'd integrate with the speech recognition
            # This is a simplified version for the web interface
        
        if st.button("Stop Voice Conversation"):
            st.info("Voice conversation stopped.")
    
    with col2:
        # Text-to-speech test
        test_text = st.text_input("Test TTS", value="Hello, this is a test of the voice synthesis system.")
        if st.button("Synthesize Speech"):
            if assistant.speech_synthesizer:
                audio_data = assistant.speech_synthesizer.synthesize(test_text)
                if audio_data:
                    st.success("Audio synthesized successfully!")
                    # In a real implementation, you'd play the audio
                else:
                    st.error("Failed to synthesize audio")
    
    # Debug information
    if st.checkbox("Show Debug Info"):
        st.header("Debug Information")
        
        # Assistant status
        status = assistant.get_status()
        st.json(status)
        
        # Vision state
        if assistant.vision_processor:
            vision_state = assistant.vision_processor.get_current_vision_state()
            st.subheader("Vision State")
            st.json(vision_state)
        
        # Personality info
        if assistant.personality_manager:
            personality_summary = assistant.personality_manager.get_personality_summary()
            st.subheader("Personality Summary")
            st.json(personality_summary)


def main():
    """Main function to run the Streamlit app."""
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.config import Config
    from core.assistant import PersonalAssistant
    
    # Initialize assistant
    config = Config()
    assistant = PersonalAssistant(config)
    
    # Create Streamlit app
    create_streamlit_app(assistant)


if __name__ == "__main__":
    main() 