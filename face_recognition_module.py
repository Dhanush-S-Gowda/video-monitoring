"""
Simplified Face Recognition Module
Uses DeepFace for face detection and a trained CNN model for face classification.
No body embedding - focuses on face recognition only.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import tensorflow as tf
from pathlib import Path

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace not available. Install with: pip install deepface")

# Constants
IMG_SIZE = (224, 224)  # Input size for the model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

class FaceRecognitionModule:
    """
    Handles face detection and recognition using trained CNN model.
    """
    
    def __init__(self, 
                 model_path: str = "final_face_model.h5",
                 labels_path: str = "class_names.txt",
                 confidence_threshold: float = 0.5,
                 detector_backend: str = "opencv"):
        """
        Initialize face recognition module.
        
        Args:
            model_path: Path to trained Keras model
            labels_path: Path to labels text file
            confidence_threshold: Confidence threshold for face recognition
            detector_backend: DeepFace detector backend (opencv, retinaface, mtcnn, ssd, dlib)
        """
        self.confidence_threshold = confidence_threshold
        self.detector_backend = detector_backend
        
        # Load trained face recognition model
        self.model = None
        self.labels = []
        
        if Path(model_path).exists():
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"Loaded face recognition model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
        else:
            print(f"Warning: Model file {model_path} not found. Train a model first using train_face_recog.py")
        
        # Load labels
        if Path(labels_path).exists():
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.labels = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(self.labels)} labels: {self.labels}")
        else:
            print(f"Warning: Labels file {labels_path} not found")
        
    def recognize_face(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Detect face in image and recognize using trained CNN.
        
        Args:
            image: BGR image (cropped person region)
            
        Returns:
            Tuple of (recognized_name, confidence)
            Returns (None, 0.0) if face not detected or not recognized
        """
        if not DEEPFACE_AVAILABLE or self.model is None:
            return None, 0.0
        
        try:
            # Extract face using DeepFace
            faces = DeepFace.extract_faces(
                img_path=image,
                enforce_detection=False,
                detector_backend=self.detector_backend,
                align=True
            )
            
            if not faces or len(faces) == 0:
                return None, 0.0
            
            # Get the first face
            face_data = faces[0]
            
            # Extract face array
            if isinstance(face_data, dict) and 'face' in face_data:
                face_img = face_data['face']
            else:
                face_img = face_data
            
            if not isinstance(face_img, np.ndarray) or face_img.size == 0:
                return None, 0.0
                
            # Convert to uint8 if needed
            if np.issubdtype(face_img.dtype, np.floating):
                face_img = (face_img * 255).clip(0, 255).astype(np.uint8)
                
            # Resize to model input size
            face_resized = cv2.resize(face_img, IMG_SIZE)
            
            # Normalize exactly as in validation
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Predict
            predictions = self.model.predict(face_batch, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            # Check threshold
            if confidence >= self.confidence_threshold:
                recognized_name = self.labels[predicted_idx] if predicted_idx < len(self.labels) else None
                return recognized_name, confidence
            else:
                return None, confidence  # Face detected but not recognized with sufficient confidence
                
        except Exception as e:
            # Silent fail - just return None
            return None, 0.0



def main():
    """
    Standalone demo of face recognition module.
    """
    recognizer = FaceRecognitionModule()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("Simplified Face Recognition Module Demo")
    print("Press 'q' to quit")
    print("-" * 50)
    
    try:
        while True:
            success, frame = cap.read()
            
            if not success:
                break
            
            # Try to recognize face in frame
            identity, confidence = recognizer.recognize_face(frame)
            
            # Display result
            if identity:
                text = f"Identity: {identity} ({confidence:.2f})"
                color = (0, 255, 0)
            else:
                text = "No face recognized"
                color = (0, 0, 255)
            
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "_main_":
    main()