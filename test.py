import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import time

# ========================
# CONFIGURATION (Match training settings)
# ========================
IMG_SIZE = (224, 224)
MODEL_PATH = 'final_face_model.h5' 
CLASS_NAMES_FILE = 'class_names.txt'
CONFIDENCE_THRESHOLD = 0.8  # Minimum prediction probability to display a name

# ========================
# 1. SETUP AND LOAD ASSETS
# ========================

def load_assets():
    """Loads the model, class names, and Haar cascade for face detection."""
    print("Loading assets...")
    
    # --- Load Model ---
    try:
        model = load_model(MODEL_PATH)
        print(f"✓ Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model from '{MODEL_PATH}'. Ensure it exists and is correct.")
        return None, None, None

    # --- Load Class Names ---
    try:
        with open(CLASS_NAMES_FILE, 'r') as f:
            class_names = [line.strip() for line in f]
        print(f"✓ {len(class_names)} class names loaded: {', '.join(class_names)}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load class names from '{CLASS_NAMES_FILE}'.")
        return None, None, None

    # --- Load Face Detector (OpenCV Haar Cascade) ---
    # This file is typically available in the opencv-python installation or online.
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print(f"❌ ERROR: Haar Cascade XML not loaded. Check the file path.")
        return None, None, None
        
    return model, class_names, face_cascade

# ========================
# 2. PREDICTION FUNCTION
# ========================

def predict_face(face_img, model, class_names):
    """Preprocesses a face image and runs prediction."""
    
    # 1. Resize and Convert
    # MobileNetV2 requires (224, 224) RGB image
    resized_img = cv2.resize(face_img, IMG_SIZE)
    
    # 2. Normalization
    # Rescale to 0-1 range (matching the ImageDataGenerator setting)
    normalized_img = resized_img / 255.0
    
    # 3. Add Batch Dimension (1, 224, 224, 3)
    input_array = np.expand_dims(normalized_img, axis=0)
    
    # 4. Predict
    predictions = model.predict(input_array, verbose=0)[0]
    
    # 5. Get Result
    max_confidence_index = np.argmax(predictions)
    max_confidence = predictions[max_confidence_index]
    
    if max_confidence >= CONFIDENCE_THRESHOLD:
        label = class_names[max_confidence_index]
        return label, max_confidence
    else:
        # If confidence is below threshold, label it 'Unknown'
        return "Unknown", max_confidence

# ========================
# 3. LIVE CAMERA LOOP
# ========================

def live_detection(model, class_names, face_cascade):
    """Opens the camera and performs live face recognition."""
    
    # Initialize camera (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera. Check camera index or permissions.")
        return
    
    print("\nStarting live face recognition. Press 'q' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection (faster)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        # (x, y) = top-left corner, (w, h) = width/height of the detected face
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region (crop)
            face_crop = frame[y:y+h, x:x+w]
            
            # Predict the identity using the Keras model
            name, confidence = predict_face(face_crop, model, class_names)
            
            # Draw bounding box (rectangle)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, Red for unknown
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Format text label
            label = f"{name}: {confidence:.2f}"
            
            # Put text label above the rectangle
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Live Face Recognition (MobileNetV2)', frame)

        # Break the loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("\nLive detection stopped.")

# ========================
# 4. MAIN EXECUTION
# ========================
if __name__ == '__main__':
    
    print("\n" + "="*60)
    print(" "*10 + "LIVE MOBILE-NET FACE RECOGNITION")
    print("="*60)
    
    model, class_names, face_cascade = load_assets()
    
    if model and class_names and face_cascade:
        live_detection(model, class_names, face_cascade)
    else:
        print("System setup failed. Cannot start live detection.")