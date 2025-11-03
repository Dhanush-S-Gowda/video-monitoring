"""
Quick Start Guide Script
Run this to get step-by-step instructions for using the system.
"""

import os
import sys
from pathlib import Path


def print_header(text):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    exists = Path(filepath).exists()
    status = "✓" if exists else "✗"
    color = "\033[92m" if exists else "\033[91m"  # Green or Red
    reset = "\033[0m"
    print(f"{color}[{status}]{reset} {description}: {filepath}")
    return exists


def main():
    print("\n" + "🚀" * 35)
    print("   PERSON TRACKING SYSTEM - QUICK START GUIDE")
    print("🚀" * 35)
    
    # Check required files
    print_header("STEP 1: Check System Files")
    
    files_status = {
        'object_tracker.py': check_file_exists('object_tracker.py', 'Object Tracker Module'),
        'face_recognition_module.py': check_file_exists('face_recognition_module.py', 'Face Recognition Module'),
        'database_manager.py': check_file_exists('database_manager.py', 'Database Manager'),
        'integrated_system.py': check_file_exists('integrated_system.py', 'Integrated System'),
        'datacollect.py': check_file_exists('datacollect.py', 'Data Collection Tool'),
        'train_face_recog.py': check_file_exists('train_face_recog.py', 'Training Script'),
        'requirements.txt': check_file_exists('requirements.txt', 'Requirements File'),
    }
    
    # Check optional files
    print("\nOptional Files:")
    has_yolo = check_file_exists('best.pt', 'YOLO Model')
    has_face_model = check_file_exists('keras_model.h5', 'Face Recognition Model')
    has_labels = check_file_exists('labels.txt', 'Labels File')
    has_images = Path('images').exists()
    
    if has_images:
        image_folders = [d for d in Path('images').iterdir() if d.is_dir()]
        print(f"\033[92m[✓]\033[0m Found {len(image_folders)} person folders in images/")
        for folder in image_folders:
            img_count = len(list(folder.glob('*.jpg'))) + len(list(folder.glob('*.png')))
            print(f"    - {folder.name}: {img_count} images")
    else:
        print(f"\033[91m[✗]\033[0m Images directory: images/")
    
    # Installation instructions
    print_header("STEP 2: Install Dependencies")
    print("Run the following command to install required packages:\n")
    print("    pip install -r requirements.txt\n")
    
    # Data collection instructions
    print_header("STEP 3: Collect Training Data")
    
    if not has_images or len(image_folders) == 0:
        print("⚠️  You need to collect face images first!\n")
    
    print("Run the data collection script for each person:")
    print("\n    python datacollect.py\n")
    print("Instructions:")
    print("  1. Enter the person's name when prompted")
    print("  2. Look at the camera from different angles")
    print("  3. Wait until 300 images are collected")
    print("  4. Repeat for each person you want to recognize")
    print("\nRecommended: Collect images for at least 2-3 people")
    
    # Training instructions
    print_header("STEP 4: Train Face Recognition Model")
    
    if not has_face_model:
        print("⚠️  Face recognition model not found!\n")
    
    print("After collecting data, train the model:")
    print("\n    python train_face_recog.py\n")
    print("This will:")
    print("  ✓ Train a CNN model on collected face images")
    print("  ✓ Save model as 'keras_model.h5'")
    print("  ✓ Generate 'labels.txt' with person names")
    print("\nTraining typically takes 5-15 minutes depending on your hardware.")
    
    # Running the system
    print_header("STEP 5: Run the Integrated System")
    
    if not has_yolo:
        print("⚠️  YOLO model not found! Download or train a YOLO11 model first.")
        print("   Quick start: The system will try to download 'yolo11n.pt' automatically\n")
    
    print("Start the complete system:")
    print("\n    python integrated_system.py\n")
    print("Optional arguments:")
    print("  --yolo-model BEST.PT       Use custom YOLO model")
    print("  --face-model MODEL.H5      Use custom face model")
    print("  --threshold 30             Re-appearance threshold (seconds)")
    print("  --camera 0                 Camera device index")
    print("\nControls:")
    print("  'q' - Quit the system")
    print("  's' - Show statistics")
    print("  'r' - Reset statistics")
    
    # Testing individual modules
    print_header("OPTIONAL: Test Individual Modules")
    print("You can test each module separately:\n")
    print("Test object tracking:")
    print("    python object_tracker.py\n")
    print("Test face recognition:")
    print("    python face_recognition_module.py\n")
    print("Test database manager:")
    print("    python database_manager.py\n")
    
    # Database queries
    print_header("BONUS: Query the Database")
    print("After running the system, you can query the presence log:\n")
    print("Using SQLite command line:")
    print("    sqlite3 presence_log.db")
    print("    SELECT * FROM presence_log WHERE date = date('now');\n")
    print("Or use the DatabaseManager class in your Python scripts.")
    
    # System readiness check
    print_header("SYSTEM READINESS CHECK")
    
    ready = all(files_status.values())
    has_training_data = has_images and len(image_folders) >= 2
    has_trained_model = has_face_model and has_labels
    
    print(f"Core modules installed:     {'✓ Yes' if ready else '✗ No'}")
    print(f"Training data collected:    {'✓ Yes' if has_training_data else '✗ No (run datacollect.py)'}")
    print(f"Model trained:              {'✓ Yes' if has_trained_model else '✗ No (run train_face_recog.py)'}")
    print(f"YOLO model available:       {'✓ Yes' if has_yolo else '⚠ Will auto-download'}")
    
    if ready and has_training_data and has_trained_model:
        print("\n\033[92m✓ System is READY to run! Execute: python integrated_system.py\033[0m")
    else:
        print("\n\033[91m✗ System is NOT ready. Follow the steps above.\033[0m")
    
    # Next steps
    print_header("WHAT HAPPENS WHEN YOU RUN THE SYSTEM?")
    print("""
1. Camera opens and starts capturing video
2. YOLO11 detects and tracks persons in the frame
3. For each tracked person:
   - System tries to recognize face using trained CNN
   - If face recognition fails, uses body embedding
   - Logs presence to SQLite database
4. Smart session management:
   - Same person re-appearing within 30 seconds → Updates existing entry
   - Same person after 30 seconds → Creates new entry
5. Database stores:
   - ID, Date, User, Start Time, End Time, Duration
6. Real-time UI shows:
   - Tracked objects with IDs
   - Recognition results
   - System statistics
    """)
    
    print("\n" + "=" * 70)
    print("  For detailed documentation, see README.md")
    print("=" * 70 + "\n")
    
    # Offer to run installer
    print("Would you like to install dependencies now? (y/n): ", end="")
    try:
        response = input().strip().lower()
        if response == 'y':
            print("\nInstalling dependencies...")
            os.system("pip install -r requirements.txt")
            print("\n✓ Installation complete!")
    except:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
