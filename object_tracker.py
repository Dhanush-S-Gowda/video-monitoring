"""
Object Tracker Module
Uses YOLO11 with built-in tracking (BoT-SORT/ByteTrack) to detect and track objects.
Outputs track IDs, bounding boxes, and cropped images for downstream processing.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import time


class ObjectTracker:
    """
    Handles object detection and tracking using YOLO11 model.
    """
    
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.5):
        """
        Initialize the object tracker.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracked_objects = {}  # track_id -> last_seen_time
        
    def process_frame(self, frame: np.ndarray, persist: bool = True) -> List[Dict]:
        """
        Process a single frame and return tracked objects.
        
        Args:
            frame: Input BGR frame from camera
            persist: Whether to persist tracks across frames
            
        Returns:
            List of dictionaries containing:
                - track_id: Unique ID for the tracked object
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score
                - class_id: Object class ID
                - class_name: Object class name
                - cropped_img: Cropped image of the detection
        """
        # Run YOLO tracking
        results = self.model.track(frame, persist=persist, conf=self.conf_threshold, verbose=False)
        
        tracked_detections = []
        current_time = time.time()
        
        if results and len(results) > 0:
            result = results[0]
            
            # Check if tracking IDs are available
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure valid coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    # Crop the region of interest
                    cropped_img = frame[y1:y2, x1:x2].copy()
                    
                    # Skip if crop is too small
                    if cropped_img.shape[0] < 20 or cropped_img.shape[1] < 20:
                        continue
                    
                    detection_data = {
                        'track_id': int(track_id),
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': result.names[cls_id],
                        'cropped_img': cropped_img,
                        'timestamp': current_time
                    }
                    
                    tracked_detections.append(detection_data)
                    self.tracked_objects[track_id] = current_time
        
        return tracked_detections
    
    def get_annotated_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and track IDs on the frame.
        
        Args:
            frame: Original frame
            detections: List of detection dictionaries
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            track_id = det['track_id']
            class_name = det['class_name']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            pass
        
        return annotated
    
    def cleanup_stale_tracks(self, timeout: float = 5.0):
        """
        Remove tracks that haven't been seen for a while.
        
        Args:
            timeout: Time in seconds after which to remove a track
        """
        current_time = time.time()
        stale_tracks = [tid for tid, last_seen in self.tracked_objects.items() 
                       if current_time - last_seen > timeout]
        
        for tid in stale_tracks:
            del self.tracked_objects[tid]


def main():
    """
    Standalone demo of object tracker.
    """
    tracker = ObjectTracker(model_path="best.pt")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("Object Tracker Demo")
    print("Press 'q' to quit")
    print("-" * 50)
    
    frame_count = 0
    
    try:
        while True:
            success, frame = cap.read()
            
            if not success:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame
            detections = tracker.process_frame(frame)
            
            # Print detection info
            if detections:
                print(f"\nFrame {frame_count}: {len(detections)} objects tracked")
                for det in detections:
                    print(f"  Track ID: {det['track_id']}, "
                          f"Class: {det['class_name']}, "
                          f"Confidence: {det['confidence']:.2f}, "
                          f"BBox: {det['bbox']}")
            
            # Visualize
            annotated_frame = tracker.get_annotated_frame(frame, detections)
            cv2.imshow("Object Tracker", annotated_frame)
            
            # Cleanup stale tracks periodically
            if frame_count % 30 == 0:
                tracker.cleanup_stale_tracks()
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
