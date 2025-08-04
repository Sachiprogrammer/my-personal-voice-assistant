"""
Object Detection Module for Spatial Awareness
"""

import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Object detection using YOLOv8 with spatial reasoning."""
    
    def __init__(self, config):
        self.config = config
        self.vision_config = config.get_vision_config()
        self.od_config = self.vision_config.get('object_detection', {})
        
        # Model settings
        self.model_name = self.od_config.get('model', 'yolov8')
        self.confidence_threshold = self.od_config.get('confidence_threshold', 0.5)
        self.nms_threshold = self.od_config.get('nms_threshold', 0.4)
        self.classes = self.od_config.get('classes', [])
        
        # Initialize YOLO model
        self.model = None
        self._load_model()
        
        # Spatial reasoning settings
        self.spatial_config = self.vision_config.get('spatial_reasoning', {})
        self.relationship_threshold = self.spatial_config.get('relationship_threshold', 0.3)
        self.max_objects = self.spatial_config.get('max_objects', 20)
    
    def _load_model(self):
        """Load YOLOv8 model for object detection."""
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            
            # Load YOLOv8 model
            if self.model_name == 'yolov8':
                self.model = YOLO('yolov8n.pt')  # Use nano model for speed
            else:
                self.model = YOLO(self.model_name)
            
            logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect objects in an image."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return []
            
            # Run detection
            results = self.model(image, conf=self.confidence_threshold, iou=self.nms_threshold)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get class and confidence
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[class_id]
                        
                        # Create detection object
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            'area': int((x2 - x1) * (y2 - y1))
                        }
                        
                        detections.append(detection)
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Limit number of objects
            detections = detections[:self.max_objects]
            
            logger.info(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    def detect_objects_cv2(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in OpenCV image array."""
        try:
            # Run detection
            results = self.model(image, conf=self.confidence_threshold, iou=self.nms_threshold)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get class and confidence
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[class_id]
                        
                        # Create detection object
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            'area': int((x2 - x1) * (y2 - y1))
                        }
                        
                        detections.append(detection)
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Limit number of objects
            detections = detections[:self.max_objects]
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting objects in image array: {e}")
            return []
    
    def analyze_spatial_relationships(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze spatial relationships between detected objects."""
        relationships = []
        
        try:
            for i, obj1 in enumerate(detections):
                for j, obj2 in enumerate(detections[i+1:], i+1):
                    relationship = self._analyze_relationship(obj1, obj2)
                    if relationship:
                        relationships.append(relationship)
            
            logger.info(f"Found {len(relationships)} spatial relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing spatial relationships: {e}")
            return []
    
    def _analyze_relationship(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze relationship between two objects."""
        try:
            bbox1 = obj1['bbox']
            bbox2 = obj2['bbox']
            center1 = obj1['center']
            center2 = obj2['center']
            
            # Calculate distances
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            # Check if objects are close enough
            if distance > 200:  # Threshold for proximity
                return None
            
            # Determine spatial relationship
            relationship_type = self._determine_relationship_type(bbox1, bbox2, center1, center2)
            
            if relationship_type:
                return {
                    'object1': obj1['class'],
                    'object2': obj2['class'],
                    'relationship': relationship_type,
                    'distance': distance,
                    'confidence': min(obj1['confidence'], obj2['confidence'])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing relationship: {e}")
            return None
    
    def _determine_relationship_type(self, bbox1: List[int], bbox2: List[int], 
                                   center1: List[int], center2: List[int]) -> Optional[str]:
        """Determine the type of spatial relationship between two objects."""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Check if one object is inside another
            if (x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2):
                return "inside"
            elif (x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1):
                return "contains"
            
            # Check horizontal relationships
            if abs(center1[1] - center2[1]) < 50:  # Similar Y coordinates
                if center1[0] < center2[0]:
                    return "to the left of"
                else:
                    return "to the right of"
            
            # Check vertical relationships
            if abs(center1[0] - center2[0]) < 50:  # Similar X coordinates
                if center1[1] < center2[1]:
                    return "above"
                else:
                    return "below"
            
            # Check diagonal relationships
            if center1[0] < center2[0] and center1[1] < center2[1]:
                return "above and to the left of"
            elif center1[0] > center2[0] and center1[1] < center2[1]:
                return "above and to the right of"
            elif center1[0] < center2[0] and center1[1] > center2[1]:
                return "below and to the left of"
            elif center1[0] > center2[0] and center1[1] > center2[1]:
                return "below and to the right of"
            
            return "near"
            
        except Exception as e:
            logger.error(f"Error determining relationship type: {e}")
            return None
    
    def get_object_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of detected objects."""
        try:
            if not detections:
                return {"message": "No objects detected"}
            
            # Count objects by class
            class_counts = {}
            for detection in detections:
                class_name = detection['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Find most prominent objects
            prominent_objects = []
            for detection in detections[:5]:  # Top 5 by confidence
                prominent_objects.append({
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'position': detection['center']
                })
            
            return {
                'total_objects': len(detections),
                'class_counts': class_counts,
                'prominent_objects': prominent_objects,
                'detections': detections
            }
            
        except Exception as e:
            logger.error(f"Error generating object summary: {e}")
            return {"error": str(e)}
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection boxes and labels on image."""
        try:
            result_image = image.copy()
            
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Draw bounding box
                cv2.rectangle(result_image, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(result_image, label, 
                           (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 255, 0), 2)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
            return image
    
    def cleanup(self):
        """Clean up resources."""
        if self.model:
            del self.model
            self.model = None
        
        logger.info("Object detector cleaned up") 