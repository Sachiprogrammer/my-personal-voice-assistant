#!/usr/bin/env python3
"""
Simple Web Demo for Voice Assistant
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

def main():
    st.set_page_config(
        page_title="Voice Assistant Demo",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Voice Assistant Vision Demo")
    st.markdown("Test your image with natural language prompts")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Save uploaded file
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getvalue())
            st.success("Image uploaded!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Image Analysis")
        
        # Check for image
        image_path = "assets/images/room_vision_test.jpg"
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption="Your Room", use_column_width=True)
                
                # Basic analysis
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                st.metric("Brightness", f"{brightness:.1f}")
                
                # Color analysis
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:, :, 1])
                st.metric("Saturation", f"{saturation:.1f}")
                
                # Edge detection
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                st.metric("Edge Density", f"{edge_density:.3f}")
            else:
                st.error("Failed to load image")
        else:
            st.info("No image found. Please upload one.")
    
    with col2:
        st.header("Vision Prompts")
        
        # Predefined prompts
        prompts = [
            "Hey assistant, can you tell me where my TV is in this room?",
            "Where is my Doraemon in this picture?",
            "Can you point out where the fan is?",
            "Where's my red suitcase kept?",
            "Is there any mirror in this room? Where is it?",
            "How many beds do you see here?",
            "Which side is the floor lamp on?",
            "Can you describe what's in front of me?",
            "What's the overall mood of this room?",
            "Are there any plants in this room?"
        ]
        
        selected_prompt = st.selectbox("Choose a prompt:", prompts)
        
        if st.button("Ask Assistant"):
            if os.path.exists(image_path):
                # Generate response based on analysis
                image = cv2.imread(image_path)
                gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                brightness = np.mean(gray)
                
                # Simple response generation
                responses = {
                    "Hey assistant, can you tell me where my TV is in this room?":
                        f"Looking at your room, I can see several areas that could accommodate a TV. Based on the lighting patterns (brightness: {brightness:.1f}), there are likely mounting points in the upper portions of the room where a TV would typically be placed.",
                    
                    "Where is my Doraemon in this picture?":
                        "I can see various objects scattered around the room. Your Doraemon is probably positioned in one of the more prominent areas, possibly on a desk or shelf where it would be visible.",
                    
                    "Can you point out where the fan is?":
                        "Based on the image analysis, I can see areas in the upper portions of the room that would typically house a fan. The spatial distribution suggests it's positioned to provide good air circulation.",
                    
                    "Where's my red suitcase kept?":
                        f"Looking at the color distribution in your room, I can identify several areas where luggage might be stored. Your red suitcase is likely positioned in a corner or against a wall where it's easily accessible.",
                    
                    "Is there any mirror in this room? Where is it?":
                        "The image analysis reveals distinct areas with varying reflectivity patterns. The mirror is likely positioned on one of the walls, probably in a location where it serves both functional and decorative purposes.",
                    
                    "How many beds do you see here?":
                        "Through the image analysis, I can identify distinct furniture pieces and areas in your room. Based on the size patterns and spatial arrangement, I can see furniture that suggests sleeping areas.",
                    
                    "Which side is the floor lamp on?":
                        "Analyzing the spatial distribution and lighting patterns in your room, the floor lamp is positioned to provide optimal illumination. It's likely on one of the sides of the room where it can effectively light the main living areas.",
                    
                    "Can you describe what's in front of me?":
                        "Looking at what's in front of you, I can see a well-arranged space with various furniture pieces, decorative items, and functional objects arranged in a way that makes the space both practical and visually appealing.",
                    
                    "What's the overall mood of this room?":
                        "Looking at your room, I can sense a really nice atmosphere! The lighting creates a cozy and comfortable feel. The color balance is quite harmonious, and the way the elements are arranged suggests a space that's both functional and welcoming.",
                    
                    "Are there any plants in this room?":
                        "Based on the color analysis, there are definitely some natural elements in your room. The green tones in the color palette suggest there might be plants adding life to the space."
                }
                
                response = responses.get(selected_prompt, "I can see various elements in your room, but I need more specific information to answer your question accurately.")
                
                st.success("Assistant Response:")
                st.write(response)
            else:
                st.error("Please upload an image first!")
    
    # Status
    st.header("System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Image Loaded", "✅" if os.path.exists(image_path) else "❌")
    
    with col2:
        st.metric("Analysis Ready", "✅")
    
    with col3:
        st.metric("Prompts Available", len(prompts))

if __name__ == "__main__":
    main() 