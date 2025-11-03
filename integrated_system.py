"""
Integrated System
Combines object tracking, face recognition, and database logging.
Main pipeline for person detection, identification, and logging.
"""

import cv2
import numpy as np
import time
import argparse
from datetime import datetime
from typing import Dict, Optional
import threading
import queue

from object_tracker import ObjectTracker
from face_recognition_module import FaceRecognitionModule
from database_manager import DatabaseManager


class IntegratedSystem:
    """
    Main system that integrates object tracking, face recognition, and logging.
    """
    
    def __init__(self, 
                 yolo_model_path: str = "best.pt",
                 face_model_path: str = "keras_model.h5",
                 labels_path: str = "labels.txt",
                 db_path: str = "presence_log.db",
                 reappear_threshold: int = 30,
                 cleanup_interval: int = 60,
                 camera_index: int = 0):
        """
        Initialize the integrated system.
        
        Args:
            yolo_model_path: Path to YOLO model
            face_model_path: Path to face recognition model
            labels_path: Path to labels file
            db_path: Path to SQLite database
            reappear_threshold: Threshold for re-appearance detection (seconds)
            cleanup_interval: Interval for cleaning up inactive tracks (seconds)
            camera_index: Camera device index
        """
        print("Initializing Integrated Person Tracking System...")
        print("=" * 60)
        
        # Initialize components
        print("Loading Object Tracker...")
        self.tracker = ObjectTracker(model_path=yolo_model_path)
        
        print("Loading Face Recognition Module...")
        self.face_recognizer = FaceRecognitionModule(
            model_path=face_model_path,
            labels_path=labels_path
        )
        
        print("Initializing Database Manager...")
        self.db = DatabaseManager(
            db_path=db_path,
            reappear_threshold=reappear_threshold
        )
        
        # Camera
        self.camera_index = camera_index
        self.cap = None
        
        # Processing settings
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'faces_recognized': 0,
            'body_matches': 0,
            'unknown': 0
        }
        
        # Threading
        self.running = False
        self.processing_queue = queue.Queue(maxsize=5)
        
        print("=" * 60)
        print("System initialized successfully!")
        print()
    
    def start_camera(self) -> bool:
        """
        Start the camera capture.
        
        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera at index {self.camera_index}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera opened successfully")
        return True
    
    def process_detections(self, detections: list):
        """
        Process detections from object tracker through face recognition and logging.
        
        Args:
            detections: List of detection dictionaries from ObjectTracker
        """
        for det in detections:
            track_id = det['track_id']
            cropped_img = det['cropped_img']
            class_name = det['class_name']
            
            # Only process person detections
            if class_name.lower() != 'person':
                continue
            
            self.stats['detections'] += 1
            
            # Run face recognition
            recognition_result = self.face_recognizer.process_detection(
                cropped_img, track_id
            )
            
            identity = recognition_result.get('identity')
            method = recognition_result.get('method')
            
            # Update statistics
            if method == 'face':
                self.stats['faces_recognized'] += 1
            elif method == 'body':
                self.stats['body_matches'] += 1
            elif method == 'unknown':
                self.stats['unknown'] += 1
            
            # Log to database if identity is known
            if identity:
                try:
                    log_id = self.db.log_detection(identity, track_id)
                    print(f"[LOG] Track {track_id} -> {identity} (method: {method}, log_id: {log_id})")
                except Exception as e:
                    print(f"Error logging detection: {e}")
            else:
                print(f"[TRACK] Track {track_id} -> Unknown person (method: {method})")
    
    def cleanup_inactive_tracks(self):
        """Periodically cleanup inactive tracks."""
        current_time = time.time()
        
        if current_time - self.last_cleanup > self.cleanup_interval:
            print("\n[CLEANUP] Running cleanup of inactive tracks...")
            self.db.cleanup_inactive_tracks(timeout=self.cleanup_interval)
            self.tracker.cleanup_stale_tracks(timeout=self.cleanup_interval)
            self.last_cleanup = current_time
    
    def draw_ui(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw UI elements on frame.
        
        Args:
            frame: Original frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        # Start with annotated frame from tracker
        annotated = self.tracker.get_annotated_frame(frame, detections)
        
        # Add system info overlay
        overlay_height = 120
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        overlay[:] = (40, 40, 40)  # Dark gray background
        
        # Statistics text
        y_offset = 25
        line_height = 22
        
        stats_text = [
            f"Frames: {self.stats['frames_processed']}  |  Detections: {self.stats['detections']}",
            f"Face Recognized: {self.stats['faces_recognized']}  |  Body Matched: {self.stats['body_matches']}  |  Unknown: {self.stats['unknown']}",
            f"Active Tracks: {len(self.tracker.tracked_objects)}  |  Time: {datetime.now().strftime('%H:%M:%S')}",
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (10, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add system name
        cv2.putText(overlay, "Person Tracking System", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, "Press 'q' to quit | 's' for stats | 'r' to reset",
                   (frame.shape[1] - 450, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine overlay with frame
        result = np.vstack([overlay, annotated])
        
        return result
    
    def print_statistics(self):
        """Print detailed statistics."""
        print("\n" + "=" * 60)
        print("SYSTEM STATISTICS")
        print("=" * 60)
        print(f"Frames Processed:    {self.stats['frames_processed']}")
        print(f"Total Detections:    {self.stats['detections']}")
        print(f"Face Recognized:     {self.stats['faces_recognized']}")
        print(f"Body Matched:        {self.stats['body_matches']}")
        print(f"Unknown:             {self.stats['unknown']}")
        print(f"Active Tracks:       {len(self.tracker.tracked_objects)}")
        print()
        
        # Show active sessions
        active_sessions = self.db.get_active_sessions()
        print(f"Active Sessions ({len(active_sessions)}):")
        for session in active_sessions:
            print(f"  {session['user']:15} | Track: {session['track_id']:3} | "
                  f"Start: {session['start_time']} | End: {session['end_time']}")
        
        # Show today's completed sessions
        today_sessions = self.db.get_sessions_by_date()
        completed = [s for s in today_sessions if s['status'] == 'completed']
        print(f"\nCompleted Sessions Today ({len(completed)}):")
        for session in completed:
            duration_min = session['duration'] // 60 if session['duration'] else 0
            duration_sec = session['duration'] % 60 if session['duration'] else 0
            print(f"  {session['user']:15} | {session['start_time']} - {session['end_time']} | "
                  f"Duration: {duration_min}m {duration_sec}s")
        
        print("=" * 60 + "\n")
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'faces_recognized': 0,
            'body_matches': 0,
            'unknown': 0
        }
        print("\n[RESET] Statistics reset")
    
    def run(self):
        """
        Main processing loop.
        """
        if not self.start_camera():
            return
        
        print("\nStarting main processing loop...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Show statistics")
        print("  'r' - Reset statistics")
        print()
        
        self.running = True
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while self.running:
                # Read frame
                success, frame = self.cap.read()
                
                if not success:
                    print("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                self.stats['frames_processed'] += 1
                fps_counter += 1
                
                # Calculate FPS every second
                if time.time() - fps_start_time > 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Track objects
                detections = self.tracker.process_frame(frame)
                
                # Process detections (face recognition + logging)
                if detections:
                    self.process_detections(detections)
                
                # Periodic cleanup
                self.cleanup_inactive_tracks()
                
                # Draw UI
                display_frame = self.draw_ui(frame, detections)
                
                # Add FPS
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Display
                cv2.imshow("Person Tracking System", display_frame)
                
                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nShutting down...")
                    break
                elif key == ord('s'):
                    self.print_statistics()
                elif key == ord('r'):
                    self.reset_statistics()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.stop()
    
    def stop(self):
        """Clean up resources."""
        self.running = False
        
        # Finalize all active sessions
        print("\nFinalizing active sessions...")
        active_sessions = self.db.get_active_sessions()
        for session in active_sessions:
            self.db.finalize_entry(session['id'])
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_statistics()
        
        print("System stopped successfully")


def main():
    """
    Entry point for the integrated system.
    """
    parser = argparse.ArgumentParser(description="Integrated Person Tracking System")
    parser.add_argument("--yolo-model", type=str, default="best.pt",
                       help="Path to YOLO model")
    parser.add_argument("--face-model", type=str, default="keras_model.h5",
                       help="Path to face recognition model")
    parser.add_argument("--labels", type=str, default="labels.txt",
                       help="Path to labels file")
    parser.add_argument("--db", type=str, default="presence_log.db",
                       help="Path to database file")
    parser.add_argument("--threshold", type=int, default=30,
                       help="Re-appearance threshold in seconds")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device index")
    parser.add_argument("--cleanup-interval", type=int, default=60,
                       help="Cleanup interval in seconds")
    
    args = parser.parse_args()
    
    # Create and run system
    system = IntegratedSystem(
        yolo_model_path=args.yolo_model,
        face_model_path=args.face_model,
        labels_path=args.labels,
        db_path=args.db,
        reappear_threshold=args.threshold,
        cleanup_interval=args.cleanup_interval,
        camera_index=args.camera
    )
    
    system.run()


if __name__ == "__main__":
    main()
