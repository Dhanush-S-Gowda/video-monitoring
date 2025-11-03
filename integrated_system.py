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
from typing import Dict
import queue

from object_tracker import ObjectTracker
from face_recognition_module import FaceRecognitionModule
from database_manager import DatabaseManager


class IntegratedSystem:
    def __init__(self,
                 yolo_model_path="best.pt",
                 face_model_path="keras_model.h5",
                 labels_path="labels.txt",
                 db_path="presence_log.db",
                 reappear_threshold=30,
                 cleanup_interval=60,
                 camera_index=0):
        print("Initializing Integrated Person Tracking System...")
        print("=" * 60)

        # Load Object Tracker
        print("Loading Object Tracker...")
        self.tracker = ObjectTracker(model_path=yolo_model_path)

        # Load Face Recognition Module
        print("Loading Face Recognition Module...")
        self.face_recognizer = FaceRecognitionModule(
            model_path=face_model_path,
            labels_path=labels_path
        )

        # Load labels (map IDs to names)
        print("Loading Labels...")
        self.labels = self._load_labels(labels_path)

        # Database Manager
        print("Initializing Database Manager...")
        self.db = DatabaseManager(db_path=db_path, reappear_threshold=reappear_threshold)

        # Camera
        self.camera_index = camera_index
        self.cap = None

        # Settings
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

        self.running = False
        self.processing_queue = queue.Queue(maxsize=5)

        print("=" * 60)
        print("System initialized successfully!\n")

    def _load_labels(self, labels_path):
        """Load label names from labels.txt"""
        try:
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(labels)} labels: {labels}")
            return labels
        except Exception as e:
            print(f"Error loading labels file: {e}")
            return []

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera at index {self.camera_index}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print("Camera opened successfully")
        return True

    def process_detections(self, detections):
        for det in detections:
            track_id = det['track_id']
            cropped_img = det['cropped_img']
            class_name = det['class_name']

            if class_name.lower() != 'person':
                continue

            self.stats['detections'] += 1

            recognition_result = self.face_recognizer.process_detection(cropped_img, track_id)
            identity = recognition_result.get('identity')
            method = recognition_result.get('method')

            if method == 'face':
                self.stats['faces_recognized'] += 1
            elif method == 'body':
                self.stats['body_matches'] += 1
            elif method == 'unknown':
                self.stats['unknown'] += 1

            # Map identity (ID) to name if found
            display_name = "Unknown"
            if identity is not None and isinstance(identity, int) and 0 <= identity < len(self.labels):
                display_name = self.labels[identity]
            elif isinstance(identity, str):
                display_name = identity

            # Log detection if identified
            if identity:
                try:
                    self.db.log_detection(display_name, track_id)
                    print(f"[LOG] Track {track_id} -> {display_name} ({method})")
                except Exception as e:
                    print(f"Error logging detection: {e}")
            else:
                print(f"[TRACK] Track {track_id} -> Unknown person ({method})")

            # Draw name on bounding box (in ObjectTracker or here)
            det['display_name'] = display_name

    def cleanup_inactive_tracks(self):
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            print("\n[CLEANUP] Cleaning inactive tracks...")
            self.db.cleanup_inactive_tracks(timeout=self.cleanup_interval)
            self.tracker.cleanup_stale_tracks(timeout=self.cleanup_interval)
            self.last_cleanup = current_time

    def draw_ui(self, frame, detections):
        annotated = self.tracker.get_annotated_frame(frame, detections)

        # Draw names above bounding boxes
        for det in detections:
            if 'bbox' in det and 'display_name' in det:
                x1, y1, x2, y2 = det['bbox']
                name = det['display_name']
                cv2.putText(annotated, name, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        overlay_height = 120
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        overlay[:] = (40, 40, 40)

        y_offset = 25
        stats_text = [
            f"Frames: {self.stats['frames_processed']} | Detections: {self.stats['detections']}",
            f"Faces: {self.stats['faces_recognized']} | Body: {self.stats['body_matches']} | Unknown: {self.stats['unknown']}",
            f"Active Tracks: {len(self.tracker.tracked_objects)} | Time: {datetime.now().strftime('%H:%M:%S')}",
        ]
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (10, y_offset + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(overlay, "Person Tracking System", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, "Press 'q' to quit | 's' for stats | 'r' to reset",
                    (frame.shape[1] - 450, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return np.vstack([overlay, annotated])

    def run(self):
        if not self.start_camera():
            return

        print("\nStarting main loop...")
        print("Controls: 'q' - Quit | 's' - Stats | 'r' - Reset\n")

        fps_counter, fps_start = 0, time.time()
        current_fps = 0
        self.running = True

        try:
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    print("Frame read failed.")
                    continue

                self.stats['frames_processed'] += 1
                fps_counter += 1

                if time.time() - fps_start > 1.0:
                    current_fps = fps_counter / (time.time() - fps_start)
                    fps_counter, fps_start = 0, time.time()

                detections = self.tracker.process_frame(frame)
                if detections:
                    self.process_detections(detections)

                self.cleanup_inactive_tracks()

                display_frame = self.draw_ui(frame, detections)
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                            (10, display_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("Person Tracking System", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Exiting system...")
                    break
                elif key == ord('s'):
                    self.print_statistics()
                elif key == ord('r'):
                    self.reset_statistics()
        finally:
            self.stop()

    def print_statistics(self):
        print("=" * 50)
        print("SYSTEM STATISTICS")
        for k, v in self.stats.items():
            print(f"{k:20}: {v}")
        print("=" * 50)

    def reset_statistics(self):
        for k in self.stats:
            self.stats[k] = 0
        print("[RESET] Statistics reset.")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("System stopped successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-model", default="best.pt")
    parser.add_argument("--face-model", default="keras_model.h5")
    parser.add_argument("--labels", default="labels.txt")
    parser.add_argument("--db", default="presence_log.db")
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cleanup-interval", type=int, default=60)
    args = parser.parse_args()

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
