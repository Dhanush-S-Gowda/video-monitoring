"""
LEGACY Object Tracking Script
This is the original simple tracking script.

For the complete modular system, use:
    - object_tracker.py (standalone tracker module)
    - integrated_system.py (full system with face recognition and logging)
"""

import cv2
from ultralytics import YOLO

print("=" * 70)
print("LEGACY OBJECT TRACKING SCRIPT")
print("=" * 70)
print("\nThis is the original simple tracking demo.")
print("For the full modular system with face recognition and logging:")
print("  1. Use: python object_tracker.py (standalone tracker)")
print("  2. Use: python integrated_system.py (complete system)")
print("\nPress 'q' to quit")
print("=" * 70 + "\n")

# Load the YOLO11 model
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Add instruction text
        cv2.putText(annotated_frame, "Legacy Tracker - Use integrated_system.py for full features", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking (Legacy)", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

print("\nFor the complete system with face recognition and database logging:")
print("  python integrated_system.py")
print("\nSee README.md for full documentation.")