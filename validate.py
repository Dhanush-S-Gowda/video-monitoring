import os
import random
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# ====================================================
# Configuration
# ====================================================
MODEL_PATH = 'final_face_model.h5'
LABELS_PATH = 'class_names.txt'
IMG_SIZE = (224, 224)
N_SAMPLES_PER_CLASS = 10 # Default number of samples to try to collect

# Get the preprocessing function used by MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def load_model_and_labels():
    """Loads the Keras model and class names."""
    try:
        # Load the trained Keras model
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print(f"✅ Model loaded from {MODEL_PATH}")

        # Load class labels
        with open(LABELS_PATH, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"✅ Labels loaded: {class_names}")
        return model, class_names
    except FileNotFoundError as e:
        print(f"❌ Error: {e.filename} not found. Ensure {MODEL_PATH} and {LABELS_PATH} exist.")
        return None, None
    except Exception as e:
        print(f"❌ An error occurred while loading the model or labels: {e}")
        return None, None

def collect_random_samples(root_folder, class_names, n_samples_per_class):
    """Collects n_samples_per_class random image paths for evaluation."""
    samples = [] # Stores (image_path, actual_class_name) tuples
    
    print("\n🔍 Collecting samples...")
    for class_name in class_names:
        class_folder = os.path.join(root_folder, class_name)
        
        if not os.path.isdir(class_folder):
            print(f"⚠️ Warning: Folder for class '{class_name}' not found at {class_folder}. Skipping.")
            continue
            
        # Get all image files (assuming jpg/png)
        files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not files:
            print(f"⚠️ Warning: No images found in {class_folder}. Skipping.")
            continue
            
        # Select n_samples_per_class random images
        n_to_select = min(n_samples_per_class, len(files))
        selected_files = random.sample(files, n_to_select)
        
        for file_path in selected_files:
            samples.append((file_path, class_name))
            
        print(f"-> Collected {n_to_select} samples for '{class_name}'.")

    return samples

def predict_on_samples(model, samples, class_names):
    """Loads images, makes predictions, and returns results."""
    results = []
    
    # Import DeepFace here to match the real system's processing
    try:
        from deepface import DeepFace
        DEEPFACE_AVAILABLE = True
    except ImportError:
        print("Warning: DeepFace not available. Install with: pip install deepface")
        DEEPFACE_AVAILABLE = False
        return []
    
    for img_path, actual_class in samples:
        try:
            # 1. Load Image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            if img is None:
                raise FileNotFoundError("Image failed to load.")

            # Convert BGR (OpenCV default) to RGB (Matplotlib default) for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 2. Extract face using DeepFace (matching real system)
            faces = DeepFace.extract_faces(
                img_path=img,
                enforce_detection=False,
                detector_backend="opencv",
                align=True
            )
            
            if not faces or len(faces) == 0:
                print(f"No face detected in {img_path}")
                continue
                
            # Get the first face
            face_data = faces[0]
            
            # Extract face array
            if isinstance(face_data, dict) and 'face' in face_data:
                face_img = face_data['face']
            else:
                face_img = face_data
            
            if not isinstance(face_img, np.ndarray) or face_img.size == 0:
                continue
                
            # Convert to uint8 if needed
            if np.issubdtype(face_img.dtype, np.floating):
                face_img = (face_img * 255).clip(0, 255).astype(np.uint8)
            
            # Resize to model input size
            face_resized = cv2.resize(face_img, IMG_SIZE)
            
            # Normalize
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # 3. Predict
            predictions = model.predict(face_batch, verbose=0)[0]
            predicted_index = np.argmax(predictions)
            confidence = predictions[predicted_index]
            
            predicted_class = class_names[predicted_index]
            
            results.append({
                'image': img_rgb,
                'actual': actual_class,
                'predicted': predicted_class,
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    return results

def display_results(results):
    """Creates a Matplotlib grid to display images and prediction details."""
    n_total = len(results)
    if n_total == 0:
        print("\n❌ No samples were successfully processed to display.")
        return

    # Dynamically determine grid size
    cols = 10
    rows = int(np.ceil(n_total / cols))
    
    plt.figure(figsize=(cols * 4, rows * 4))
    plt.suptitle("Model Prediction Results", fontsize=16)

    for i, res in enumerate(results):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(res['image'])
        plt.axis('off')

        actual = res['actual']
        predicted = res['predicted']
        conf = res['confidence']
        
        # Color the text based on correctness
        is_correct = (actual == predicted)
        color = 'green' if is_correct else 'red'
        
        title = f"Actual: {actual}\nPred: {predicted} ({conf*100:.1f}%)"
        
        plt.title(title, color=color, fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit suptitle
    plt.show()

# ====================================================
# Main Execution
# ====================================================

if __name__ == "__main__":
    # 1. Load Model and Labels
    model, class_names = load_model_and_labels()
    if model is None:
        exit()
        
    # Determine how many samples to collect (avoids the 'global' error)
    samples_to_collect = N_SAMPLES_PER_CLASS 
    
    


    # 2. Get Input Path
    folder_path = input("\nEnter the path to the main image folder (e.g., 'images/'): ").strip()
    if not os.path.isdir(folder_path):
        print(f"❌ Error: Folder not found at '{folder_path}'. Exiting.")
        exit()

    # 3. Collect Samples (Pass the locally determined count)
    samples = collect_random_samples(folder_path, class_names, samples_to_collect)
    
    if not samples:
        print("❌ No valid samples were collected. Exiting.")
        exit()

    # 4. Predict
    results = predict_on_samples(model, samples, class_names)

    # 5. Display
    display_results(results)

    print("\nScript finished.")