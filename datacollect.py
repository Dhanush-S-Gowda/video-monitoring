import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from deepface import DeepFace
except Exception as e:
    print("DeepFace import failed:", e)
    print("Install with: pip install deepface")
    sys.exit(1)

# Configuration from environment
NAME = input("Enter name of the user: ").strip() or "user"
TARGET_COUNT = 300
FRAME_SKIP = 1  # Process every frame for better detection
DETECTOR_BACKEND = "opencv"  # opencv is fastest and works well for frontal faces
DEBUG = True  # Show detection info
OUT_DIR = Path("images") / NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open default camera (index 0). Exiting.")
    sys.exit(1)

count = len(list(OUT_DIR.glob("*.jpg")))
print(f"Saving to {OUT_DIR} — starting from {count}. Target: {TARGET_COUNT}")
frame_idx = 0

# Setup window first
cv2.namedWindow('Face Collector', cv2.WINDOW_NORMAL)
# Set a reasonable size - will be resizable
cv2.resizeWindow('Face Collector', 800, 600)

try:
    while count < TARGET_COUNT:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, retrying in 0.5s...")
            time.sleep(0.5)
            continue

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        try:
            # DeepFace expects the image in img_path even for numpy arrays
            faces = DeepFace.extract_faces(
                img_path=frame,  # Pass frame directly as img_path
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,  # Require face detection
                align=True,
                expand_percentage=20  # Expand detection area slightly
            )
        except Exception as ex:
            if DEBUG:
                print("DeepFace.extract_faces raised:", ex)
            faces = []

        face_img = None
        bbox = None
        if faces:
            first = faces[0]
            if isinstance(first, dict) and 'face' in first:
                f = first['face']
            else:
                f = first

            if isinstance(f, np.ndarray) and f.size > 0:
                face_img = f
                if np.issubdtype(face_img.dtype, np.floating):
                    face_img = (face_img * 255).clip(0, 255).astype(np.uint8)
                elif face_img.dtype != np.uint8:
                    face_img = face_img.astype(np.uint8)

                # Convert RGB->BGR if it appears to be RGB
                if face_img.ndim == 3 and face_img.shape[2] == 3:
                    if np.mean(face_img[:, :, 0]) > np.mean(face_img[:, :, 2]) + 1:
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                if face_img.ndim == 2:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)

                if DEBUG:
                    print(f"Extracted face shape={face_img.shape} dtype={face_img.dtype}")
                # Try to get bounding box info if available
                if isinstance(first, dict) and 'facial_area' in first:
                    fa = first.get('facial_area')
                    if isinstance(fa, dict):
                        # DeepFace uses different keys on versions: try both sets
                        x = int(fa.get('x', fa.get('left', 0)))
                        y = int(fa.get('y', fa.get('top', 0)))
                        w = int(fa.get('w', fa.get('width', 0)))
                        h = int(fa.get('h', fa.get('height', 0)))
                        bbox = (x, y, w, h)

        if face_img is not None:
            save_path = OUT_DIR / f"{NAME}_{count+1:04d}.jpg"
            success = cv2.imwrite(str(save_path), face_img)
            if success:
                count += 1
                print(f"Saved {save_path} ({count}/{TARGET_COUNT})")
            else:
                print(f"Failed to write image to {save_path}")

        # Annotate full-screen preview with bounding box (if any) and overlay text
        display_frame = frame.copy()
        # draw bbox if available
        if bbox is not None:
            x, y, w, h = bbox
            # clamp bbox to frame
            x = max(0, x)
            y = max(0, y)
            w = max(0, min(w, display_frame.shape[1] - x))
            h = max(0, min(h, display_frame.shape[0] - y))
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        elif DEBUG:
            cv2.putText(display_frame, "No face detected", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        # overlay name and count
        text = f"{NAME}  {count}/{TARGET_COUNT}"
        # choose font scale based on width
        fw = display_frame.shape[1]
        font_scale = 1.2 if fw >= 1280 else 0.9
        thickness = 2 if fw >= 1280 else 1
        cv2.putText(display_frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        # show helpful hint
        hint = "Press 'q' to quit"
        cv2.putText(display_frame, hint, (30, display_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Face Collector', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit.")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Collected {count} images to {OUT_DIR}")
