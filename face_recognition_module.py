"""
Face Recognition Module
Uses DeepFace for face detection and a trained CNN model for face classification.
Also handles body embedding extraction for fallback identification.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import tensorflow as tf
from pathlib import Path
import os

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace not available. Install with: pip install deepface")


class FaceRecognitionModule:
    """
    Handles face detection, recognition, and body embedding extraction.
    """
    
    def __init__(self, 
                 model_path: str = "keras_model.h5",
                 labels_path: str = "labels.txt",
                 face_threshold: float = 0.5,
                 body_threshold: float = 0.7,
                 detector_backend: str = "opencv"):
        """
        Initialize face recognition module.
        
        Args:
            model_path: Path to trained Keras model
            labels_path: Path to labels text file
            face_threshold: Confidence threshold for face recognition
            body_threshold: Similarity threshold for body embedding matching
            detector_backend: DeepFace detector backend (opencv, retinaface, mtcnn, ssd, dlib)
        """
        self.face_threshold = face_threshold
        self.body_threshold = body_threshold
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
        
        # Body embedding database (track_id -> embedding)
        self.body_embeddings_db = {}
        
        # Face embedding model for similarity matching
        self.face_embedding_model = "Facenet512"  # DeepFace model for embeddings
        
    def detect_and_recognize_face(self, image: np.ndarray) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Detect face in image and recognize using trained CNN.
        
        Args:
            image: BGR image (cropped person region)
            
        Returns:
            Tuple of (face_detected, recognized_name, confidence)
        """
        if not DEEPFACE_AVAILABLE or self.model is None:
            return False, None, None
        
        try:
            # Extract face using DeepFace
            faces = DeepFace.extract_faces(
                img_path=image,
                enforce_detection=False,
                detector_backend=self.detector_backend,
                align=True
            )
            
            if not faces or len(faces) == 0:
                return False, None, None
            
            # Get the first face
            face_data = faces[0]
            
            # Extract face array
            if isinstance(face_data, dict) and 'face' in face_data:
                face_img = face_data['face']
            else:
                face_img = face_data
            
            if not isinstance(face_img, np.ndarray) or face_img.size == 0:
                return False, None, None
            
            # Convert to uint8 if needed
            if np.issubdtype(face_img.dtype, np.floating):
                face_img = (face_img * 255).clip(0, 255).astype(np.uint8)
            
            # Ensure BGR format
            if face_img.ndim == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            elif face_img.shape[2] == 3:
                # Check if RGB (DeepFace returns RGB)
                if np.mean(face_img[:, :, 0]) > np.mean(face_img[:, :, 2]) + 1:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            
            # Resize to model input size
            input_size = (224, 224)  # Match training size
            face_resized = cv2.resize(face_img, input_size)
            
            # Normalize
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Predict
            predictions = self.model.predict(face_batch, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            # Check threshold
            if confidence >= self.face_threshold:
                recognized_name = self.labels[predicted_idx] if predicted_idx < len(self.labels) else "Unknown"
                return True, recognized_name, confidence
            else:
                return True, None, confidence  # Face detected but not recognized
                
        except Exception as e:
            print(f"Face recognition error: {e}")
            return False, None, None
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using DeepFace for similarity comparison.
        
        Args:
            image: BGR image
            
        Returns:
            Face embedding vector or None
        """
        if not DEEPFACE_AVAILABLE:
            return None
        
        try:
            # Use DeepFace to get embeddings
            embedding_objs = DeepFace.represent(
                img_path=image,
                model_name=self.face_embedding_model,
                enforce_detection=False,
                detector_backend=self.detector_backend
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]['embedding'])
                return embedding
            
        except Exception as e:
            print(f"Face embedding extraction error: {e}")
        
        return None
    
    def extract_body_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract body embedding using a simple CNN approach.
        For better results, use a dedicated person re-identification model.
        
        Args:
            image: BGR image of person
            
        Returns:
            Body embedding vector
        """
        try:
            # Resize to standard size
            resized = cv2.resize(image, (128, 256))
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            # Extract color histogram features
            hist_features = []
            for i in range(3):  # BGR channels
                hist = cv2.calcHist([resized], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)
            
            # Extract HOG features (simplified)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Create simple feature vector
            embedding = np.array(hist_features, dtype=np.float32)
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"Body embedding extraction error: {e}")
            # Return zero vector on error
            return np.zeros(96, dtype=np.float32)
    
    def match_body_embedding(self, embedding: np.ndarray, track_id: int) -> Tuple[Optional[str], float]:
        """
        Match body embedding against stored database using cosine similarity.
        
        Args:
            embedding: Body embedding vector
            track_id: Current track ID
            
        Returns:
            Tuple of (matched_identity, similarity_score)
        """
        if len(self.body_embeddings_db) == 0:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for stored_id, stored_data in self.body_embeddings_db.items():
            if stored_id == track_id:
                continue  # Skip self
            
            stored_embedding = stored_data['embedding']
            
            # Cosine similarity
            similarity = np.dot(embedding, stored_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_embedding) + 1e-8
            )
            
            if similarity > best_similarity:
                best_similarity = float(similarity)
                best_match = stored_data.get('identity')
        
        if best_similarity >= self.body_threshold:
            return best_match, best_similarity
        
        return None, best_similarity
    
    def store_body_embedding(self, track_id: int, embedding: np.ndarray, identity: Optional[str] = None):
        """
        Store body embedding for a track ID.
        
        Args:
            track_id: Track ID
            embedding: Body embedding vector
            identity: Associated identity (if known)
        """
        self.body_embeddings_db[track_id] = {
            'embedding': embedding,
            'identity': identity
        }
    
    def process_detection(self, cropped_img: np.ndarray, track_id: int) -> Dict:
        """
        Process a cropped detection image.
        First tries face recognition, then falls back to body embedding matching.
        
        Args:
            cropped_img: Cropped BGR image of detected person
            track_id: Track ID from object tracker
            
        Returns:
            Dictionary with recognition results:
                - face_detected: bool
                - identity: str or None
                - confidence: float or None
                - method: 'face' or 'body' or 'unknown'
        """
        result = {
            'face_detected': False,
            'identity': None,
            'confidence': None,
            'method': 'unknown'
        }
        
        # Try face recognition first
        face_detected, recognized_name, face_confidence = self.detect_and_recognize_face(cropped_img)
        
        if face_detected and recognized_name:
            # Face recognized successfully
            result['face_detected'] = True
            result['identity'] = recognized_name
            result['confidence'] = face_confidence
            result['method'] = 'face'
            
            # Also store body embedding for this identity
            body_embedding = self.extract_body_embedding(cropped_img)
            self.store_body_embedding(track_id, body_embedding, recognized_name)
            
        else:
            # Face not recognized, try body embedding
            body_embedding = self.extract_body_embedding(cropped_img)
            matched_identity, similarity = self.match_body_embedding(body_embedding, track_id)
            
            if matched_identity:
                # Body matched to known identity
                result['identity'] = matched_identity
                result['confidence'] = similarity
                result['method'] = 'body'
            else:
                # New person, store embedding without identity
                self.store_body_embedding(track_id, body_embedding, None)
                result['method'] = 'unknown'
            
            if face_detected:
                result['face_detected'] = True
        
        return result


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
    
    print("Face Recognition Module Demo")
    print("Press 'q' to quit")
    print("-" * 50)
    
    try:
        while True:
            success, frame = cap.read()
            
            if not success:
                break
            
            # Simulate a detection (use full frame for demo)
            result = recognizer.process_detection(frame, track_id=1)
            
            # Display result
            text = f"Method: {result['method']}"
            if result['identity']:
                text += f" | Identity: {result['identity']}"
                if result['confidence']:
                    text += f" ({result['confidence']:.2f})"
            
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
