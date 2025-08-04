"""
Vision Processor Module for Real-time Spatial Awareness
"""

import logging
import cv2
import numpy as np
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class VisionProcessor:
    """Real-time vision processing with spatial awareness."""
    
    def __init__(self, config):
        self.config = config
        self.vision_config = config.get_vision_config()
        self.camera_config = self.vision_config.get('camera', {})
        
        # Camera settings
        self.device_id = self.camera_config.get('device_id', 0)
        self.resolution = self.camera_config.get('resolution', [640, 480])
        self.fps = self.camera_config.get('fps', 30)
        
        # Processing settings
        self.processing_interval = 1.0  # Process every second
        self.is_processing = False
        self.is_camera_active = False
        
        # Camera and processing threads
        self.camera_thread = None
        self.processing_thread = None
        
        # Camera capture
        self.cap = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Object detector (will be set by assistant)
        self.object_detector = None
        
        # Current vision state
        self.current_detections = []
        self.current_relationships = []
        self.last_processed_time = 0
        
        # Callbacks
        self.on_detection_update: Optional[Callable] = None
        self.on_relationship_update: Optional[Callable] = None
    
    def set_object_detector(self, object_detector):
        """Set the object detector instance."""
        self.object_detector = object_detector
    
    def start_camera(self) -> bool:
        """Start camera capture."""
        try:
            if self.is_camera_active:
                logger.warning("Camera is already active")
                return True
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera device {self.device_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Start camera thread
            self.is_camera_active = True
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            
            logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture."""
        self.is_camera_active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        
        logger.info("Camera stopped")
    
    def _camera_loop(self):
        """Camera capture loop."""
        while self.is_camera_active:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                else:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in camera loop: {e}")
                time.sleep(0.1)
    
    def start_processing(self) -> bool:
        """Start vision processing."""
        if not self.object_detector:
            logger.error("Object detector not set")
            return False
        
        if not self.is_camera_active:
            logger.error("Camera not active")
            return False
        
        if self.is_processing:
            logger.warning("Processing is already active")
            return True
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Vision processing started")
        return True
    
    def stop_processing(self):
        """Stop vision processing."""
        self.is_processing = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Vision processing stopped")
    
    def _processing_loop(self):
        """Vision processing loop."""
        while self.is_processing:
            try:
                current_time = time.time()
                
                # Process at specified interval
                if current_time - self.last_processed_time >= self.processing_interval:
                    self._process_current_frame()
                    self.last_processed_time = current_time
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _process_current_frame(self):
        """Process the current camera frame."""
        try:
            with self.frame_lock:
                if self.current_frame is None:
                    return
                frame = self.current_frame.copy()
            
            # Detect objects
            detections = self.object_detector.detect_objects_cv2(frame)
            
            # Analyze spatial relationships
            relationships = self.object_detector.analyze_spatial_relationships(detections)
            
            # Update current state
            self.current_detections = detections
            self.current_relationships = relationships
            
            # Call callbacks if set
            if self.on_detection_update:
                self.on_detection_update(detections)
            
            if self.on_relationship_update:
                self.on_relationship_update(relationships)
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def get_current_vision_state(self) -> Dict[str, Any]:
        """Get current vision state."""
        return {
            'detections': self.current_detections,
            'relationships': self.current_relationships,
            'last_processed': self.last_processed_time,
            'camera_active': self.is_camera_active,
            'processing_active': self.is_processing
        }
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze a single image."""
        try:
            if not self.object_detector:
                return {"error": "Object detector not available"}
            
            # Detect objects
            detections = self.object_detector.detect_objects(image_path)
            
            # Analyze relationships
            relationships = self.object_detector.analyze_spatial_relationships(detections)
            
            # Generate summary
            summary = self.object_detector.get_object_summary(detections)
            
            return {
                'detections': detections,
                'relationships': relationships,
                'summary': summary,
                'image_path': image_path
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": str(e)}
    
    def generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a natural language summary of vision analysis."""
        try:
            detections = analysis.get('detections', [])
            relationships = analysis.get('relationships', [])
            
            if not detections:
                return "I don't see any objects in the image."
            
            # Count objects
            object_counts = {}
            for detection in detections:
                obj_class = detection['class']
                object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
            
            # Generate summary
            summary_parts = []
            
            # Object count summary
            if len(object_counts) == 1:
                obj_name, count = list(object_counts.items())[0]
                summary_parts.append(f"I can see {count} {obj_name}")
                if count > 1:
                    summary_parts[-1] += "s"
            else:
                obj_list = []
                for obj_name, count in object_counts.items():
                    if count == 1:
                        obj_list.append(f"a {obj_name}")
                    else:
                        obj_list.append(f"{count} {obj_name}s")
                
                summary_parts.append(f"I can see {', '.join(obj_list)}")
            
            # Relationship summary
            if relationships:
                rel_summaries = []
                for rel in relationships[:3]:  # Limit to 3 relationships
                    rel_summary = f"the {rel['object1']} is {rel['relationship']} the {rel['object2']}"
                    rel_summaries.append(rel_summary)
                
                if rel_summaries:
                    summary_parts.append(f"Regarding spatial relationships: {', '.join(rel_summaries)}.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "I'm having trouble analyzing the image."
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture current frame from camera."""
        try:
            with self.frame_lock:
                if self.current_frame is not None:
                    return self.current_frame.copy()
                return None
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def save_frame(self, output_path: str) -> bool:
        """Save current frame to file."""
        try:
            frame = self.capture_frame()
            if frame is not None:
                cv2.imwrite(output_path, frame)
                logger.info(f"Frame saved to: {output_path}")
                return True
            else:
                logger.warning("No frame available to save")
                return False
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        if not self.cap:
            return {"error": "Camera not initialized"}
        
        try:
            return {
                "device_id": self.device_id,
                "resolution": self.resolution,
                "fps": self.fps,
                "is_opened": self.cap.isOpened(),
                "frame_width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "frame_height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "actual_fps": self.cap.get(cv2.CAP_PROP_FPS)
            }
        except Exception as e:
            logger.error(f"Error getting camera info: {e}")
            return {"error": str(e)}
    
    def set_processing_interval(self, interval: float):
        """Set the processing interval in seconds."""
        self.processing_interval = max(0.1, interval)  # Minimum 0.1 seconds
        logger.info(f"Processing interval set to {self.processing_interval} seconds")
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_processing()
        self.stop_camera()
        
        logger.info("Vision processor cleaned up") 