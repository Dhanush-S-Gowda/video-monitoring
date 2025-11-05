"""
Simplified Main System
Receives object tracking data from Raspberry Pi and performs face recognition.
Maps track IDs to face predictions and logs entry/exit times in database.
"""

import cv2
import numpy as np
import socket
import struct
import pickle
import time
from datetime import datetime
from typing import Dict, Optional

from face_recognition_module import FaceRecognitionModule
from database_manager import DatabaseManager


class SimplifiedSystem:
    def __init__(self,
                 face_model_path="final_face_model.h5",
                 labels_path="class_names.txt",
                 db_path="presence_log.db",
                 rpi_host="10.125.32.71",
                 rpi_port=8000):
        """
        Initialize the simplified tracking system.
        
        Args:
            face_model_path: Path to trained face recognition model
            labels_path: Path to labels file
            db_path: Path to SQLite database
            rpi_host: Raspberry Pi IP address
            rpi_port: Raspberry Pi port
        """
        print("Initializing Simplified Person Tracking System...")
        print("=" * 60)

        # Load Face Recognition Module
        print("Loading Face Recognition Module...")
        self.face_recognizer = FaceRecognitionModule(
            model_path=face_model_path,
            labels_path=labels_path
        )

        # Database Manager
        print("Initializing Database Manager...")
        self.db = DatabaseManager(db_path=db_path)

        # Raspberry Pi connection
        self.rpi_host = rpi_host
        self.rpi_port = rpi_port
        self.client_socket = None
        self.data = b""
        self.payload_size = struct.calcsize("Q")

        # Track ID to identity mapping
        # {track_id: {'identity': str, 'confidence': float, 'entry_id': int}}
        self.track_mapping = {}

        # Track IDs seen in current frame
        self.current_tracks = set()
        self.previous_tracks = set()

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'face_recognitions': 0,
            'guests': 0,
            'total_entries': 0,
            'total_exits': 0
        }

        self.running = False

        print("=" * 60)
        print("System initialized successfully!\n")

    def connect_to_rpi(self):
        """Connect to Raspberry Pi stream."""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.rpi_host, self.rpi_port))
            print(f"[INFO] Connected to Raspberry Pi at {self.rpi_host}:{self.rpi_port}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect to Raspberry Pi: {e}")
            return False

    def receive_frame_data(self):
        """
        Receive frame and tracking data from Raspberry Pi.
        
        Returns:
            Tuple of (frame, boxes) or (None, None) if failed
        """
        try:
            # Retrieve message size
            while len(self.data) < self.payload_size:
                packet = self.client_socket.recv(4096)
                if not packet:
                    return None, None
                self.data += packet

            packed_msg_size = self.data[:self.payload_size]
            self.data = self.data[self.payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            # Retrieve full message data
            while len(self.data) < msg_size:
                self.data += self.client_socket.recv(4096)

            frame_data = self.data[:msg_size]
            self.data = self.data[msg_size:]

            # Deserialize
            payload = pickle.loads(frame_data)
            frame = cv2.imdecode(payload["frame"], cv2.IMREAD_COLOR)
            boxes = payload["boxes"]

            return frame, boxes

        except Exception as e:
            print(f"[ERROR] Failed to receive frame data: {e}")
            return None, None

    def process_detection(self, cropped_img, track_id):
        """
        Process a detection and assign identity using face recognition.
        
        Args:
            cropped_img: Cropped image of the person
            track_id: Track ID from YOLO tracker
        """
        # Check if we already have a high-confidence assignment for this track
        if track_id in self.track_mapping:
            existing = self.track_mapping[track_id]
            # If already assigned and not a guest, try to improve confidence
            if existing['identity'] != "Guest":
                # Try face recognition to see if we get better confidence
                identity, confidence = self.face_recognizer.recognize_face(cropped_img)
                
                # Only update if new confidence is significantly better
                if identity and confidence > existing['confidence']:
                    print(f"[UPDATE] Track {track_id}: {existing['identity']} ({existing['confidence']:.2f}) -> {identity} ({confidence:.2f})")
                    self.track_mapping[track_id]['identity'] = identity
                    self.track_mapping[track_id]['confidence'] = confidence
                    
                    # Update database entry
                    self.db.update_identity(existing['entry_id'], identity)
                return

            # If it's a guest, try to recognize
            if existing['identity'] == "Guest":
                identity, confidence = self.face_recognizer.recognize_face(cropped_img)
                if identity:
                    print(f"[UPDATE] Track {track_id}: Guest -> {identity} ({confidence:.2f})")
                    self.track_mapping[track_id]['identity'] = identity
                    self.track_mapping[track_id]['confidence'] = confidence
                    
                    # Update database entry
                    self.db.update_identity(existing['entry_id'], identity)
                    self.stats['face_recognitions'] += 1
                return

        # New track - try to recognize face
        identity, confidence = self.face_recognizer.recognize_face(cropped_img)

        if not identity:
            identity = "Guest"
            confidence = 0.0
            self.stats['guests'] += 1
        else:
            self.stats['face_recognitions'] += 1

        # Log entry in database
        entry_id = self.db.log_entry(identity, track_id)

        # Store mapping
        self.track_mapping[track_id] = {
            'identity': identity,
            'confidence': confidence,
            'entry_id': entry_id
        }

        self.stats['total_entries'] += 1
        print(f"[NEW] Track {track_id} -> {identity} (confidence: {confidence:.2f}) | Entry ID: {entry_id}")

    def process_frame(self, frame, boxes):
        """
        Process received frame and bounding boxes.
        
        Args:
            frame: BGR image from camera
            boxes: List of detection boxes from YOLO tracker
        """
        self.stats['frames_processed'] += 1
        self.current_tracks = set()

        # Process each detection
        for box in boxes:
            track_id = box["track_id"]
            bbox = box["bbox"]
            x1, y1, x2, y2 = bbox

            self.current_tracks.add(track_id)
            self.stats['total_detections'] += 1

            # Crop person region
            cropped_img = frame[y1:y2, x1:x2].copy()

            # Skip if crop is too small
            if cropped_img.shape[0] < 20 or cropped_img.shape[1] < 20:
                continue

            # Process detection (face recognition and assignment)
            self.process_detection(cropped_img, track_id)

        # Detect lost tracks (exits)
        lost_tracks = self.previous_tracks - self.current_tracks

        for track_id in lost_tracks:
            if track_id in self.track_mapping:
                identity = self.track_mapping[track_id]['identity']
                entry_id = self.track_mapping[track_id]['entry_id']

                # Log exit
                self.db.log_exit(entry_id)

                print(f"[EXIT] Track {track_id} ({identity}) | Entry ID: {entry_id}")
                
                # Remove from mapping
                del self.track_mapping[track_id]
                self.stats['total_exits'] += 1

        # Update previous tracks
        self.previous_tracks = self.current_tracks.copy()

    def draw_ui(self, frame, boxes):
        """
        Draw UI overlay on frame.
        
        Args:
            frame: BGR image
            boxes: List of detection boxes
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw bounding boxes with identities
        for box in boxes:
            track_id = box["track_id"]
            bbox = box["bbox"]
            x1, y1, x2, y2 = bbox
            conf = box["confidence"]

            # Get identity for this track
            identity = "Unknown"
            if track_id in self.track_mapping:
                identity = self.track_mapping[track_id]['identity']

            # Draw box
            color = (0, 255, 0) if identity != "Guest" else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{identity} | ID:{track_id} | {conf:.2f}"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw stats overlay
        overlay_height = 130
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        overlay[:] = (40, 40, 40)

        y_offset = 25
        stats_text = [
            f"Frames: {self.stats['frames_processed']} | Detections: {self.stats['total_detections']}",
            f"Face Recognized: {self.stats['face_recognitions']} | Guests: {self.stats['guests']}",
            f"Entries: {self.stats['total_entries']} | Exits: {self.stats['total_exits']}",
            f"Active Tracks: {len(self.track_mapping)} | Time: {datetime.now().strftime('%H:%M:%S')}",
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(overlay, "Simplified Person Tracking System", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, "Press 'q' to quit | 's' for stats | 'r' to reset",
                   (frame.shape[1] - 450, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return np.vstack([overlay, annotated])

    def print_statistics(self):
        """Print system statistics."""
        print("\n" + "=" * 60)
        print("SYSTEM STATISTICS")
        print("=" * 60)
        for k, v in self.stats.items():
            print(f"{k:25}: {v}")
        print("=" * 60 + "\n")

    def reset_statistics(self):
        """Reset statistics counters."""
        for k in self.stats:
            self.stats[k] = 0
        print("[RESET] Statistics reset.")

    def run(self):
        """Main loop."""
        if not self.connect_to_rpi():
            return

        print("\nStarting main loop...")
        print("Controls: 'q' - Quit | 's' - Stats | 'r' - Reset\n")

        fps_counter = 0
        fps_start = time.time()
        current_fps = 0
        self.running = True

        try:
            while self.running:
                # Receive frame and boxes from Raspberry Pi
                frame, boxes = self.receive_frame_data()

                if frame is None or boxes is None:
                    print("[ERROR] Failed to receive frame. Retrying...")
                    time.sleep(1)
                    continue

                fps_counter += 1

                if time.time() - fps_start > 1.0:
                    current_fps = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()

                # Process frame
                self.process_frame(frame, boxes)

                # Draw UI
                display_frame = self.draw_ui(frame, boxes)
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                           (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("Person Tracking System", display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nExiting system...")
                    break
                elif key == ord('s'):
                    self.print_statistics()
                elif key == ord('r'):
                    self.reset_statistics()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] {e}")
        finally:
            self.stop()

    def stop(self):
        """Clean up and stop system."""
        self.running = False
        
        # Mark any remaining active tracks as exited
        for track_id, info in list(self.track_mapping.items()):
            entry_id = info['entry_id']
            identity = info['identity']
            self.db.log_exit(entry_id)
            print(f"[EXIT] Track {track_id} ({identity}) | Entry ID: {entry_id}")

        if self.client_socket:
            self.client_socket.close()
        
        cv2.destroyAllWindows()
        print("System stopped successfully.")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Simplified Person Tracking System")
    parser.add_argument("--face-model", default="final_face_model.h5", help="Path to face recognition model")
    parser.add_argument("--labels", default="class_names.txt", help="Path to labels file")
    parser.add_argument("--db", default="presence_log.db", help="Path to database file")
    parser.add_argument("--rpi-host", default="10.125.32.71", help="Raspberry Pi IP address")
    parser.add_argument("--rpi-port", type=int, default=8000, help="Raspberry Pi port")
    
    args = parser.parse_args()

    system = SimplifiedSystem(
        face_model_path=args.face_model,
        labels_path=args.labels,
        db_path=args.db,
        rpi_host=args.rpi_host,
        rpi_port=args.rpi_port
    )
    
    system.run()


if __name__ == "__main__":
    main()
