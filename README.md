# Person Tracking and Recognition System

A modular system for real-time person detection, tracking, face recognition, and presence logging using YOLO11, DeepFace, and custom CNN models.

## 🏗️ Architecture

The system is composed of four main modules:

1. **Object Tracker** (`object_tracker.py`) - Uses YOLO11 with BoT-SORT/ByteTrack for person detection and tracking
2. **Face Recognition Module** (`face_recognition_module.py`) - Handles face detection/recognition using DeepFace and trained CNN
3. **Database Manager** (`database_manager.py`) - SQLite-based logging with smart re-appearance detection
4. **Integrated System** (`integrated_system.py`) - Main pipeline connecting all modules

## 📋 Features

- ✅ Real-time person detection and tracking with persistent track IDs
- ✅ Face recognition using custom trained CNN model
- ✅ Body embedding fallback when face is not recognized
- ✅ Smart re-appearance detection (30-second threshold configurable)
- ✅ SQLite database for presence logging
- ✅ Automatic session merging for re-appearances
- ✅ Statistics and monitoring UI
- ✅ Modular design - each component can run standalone

## 🚀 Quick Start

### 1. Install Dependencies

```cmd
pip install -r requirements.txt
```

### 2. Collect Training Data

Collect face images for each person you want to recognize:

```cmd
python datacollect.py
```

- Enter the person's name when prompted
- Look at the camera from different angles
- The script will collect 300 images per person
- Images are saved in `images/<person_name>/`

Repeat for each person.

### 3. Train Face Recognition Model

After collecting data for all people:

```cmd
python train_face_recog.py
```

This will:
- Train a CNN model on the collected face images
- Save the model as `keras_model.h5`
- Generate `labels.txt` with person names

### 4. Run the Integrated System

```cmd
python integrated_system.py
```

Optional arguments:
```cmd
python integrated_system.py --yolo-model best.pt --face-model keras_model.h5 --threshold 30 --camera 0
```

## 📊 Database Schema

### `presence_log` Table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| date | TEXT | Date (YYYY-MM-DD) |
| user | TEXT | Person's name |
| start_time | TEXT | Session start time (HH:MM:SS) |
| end_time | TEXT | Session end time (HH:MM:SS) |
| duration | INTEGER | Duration in seconds |
| track_id | INTEGER | Associated track ID |
| status | TEXT | 'active' or 'completed' |
| created_at | TEXT | Timestamp |
| updated_at | TEXT | Timestamp |

### `track_mapping` Table

Maps track IDs to users and their current presence log entry.

## 🔧 Module Details

### Object Tracker (`object_tracker.py`)

**Purpose**: Detect and track persons in video frames

**Key Features**:
- YOLO11 with built-in tracking (BoT-SORT/ByteTrack)
- Returns track IDs, bounding boxes, and cropped images
- Automatic cleanup of stale tracks

**Standalone Usage**:
```python
from object_tracker import ObjectTracker

tracker = ObjectTracker(model_path="best.pt")
detections = tracker.process_frame(frame)
```

**Output Format**:
```python
{
    'track_id': 1,
    'bbox': [x1, y1, x2, y2],
    'confidence': 0.95,
    'class_id': 0,
    'class_name': 'person',
    'cropped_img': numpy_array,
    'timestamp': 1234567890.123
}
```

### Face Recognition Module (`face_recognition_module.py`)

**Purpose**: Identify persons using face recognition and body embeddings

**Key Features**:
- Face detection using DeepFace
- Face classification using trained CNN
- Body embedding extraction for fallback
- Cosine similarity matching

**Standalone Usage**:
```python
from face_recognition_module import FaceRecognitionModule

recognizer = FaceRecognitionModule()
result = recognizer.process_detection(cropped_img, track_id)
```

**Output Format**:
```python
{
    'face_detected': True,
    'identity': 'John',
    'confidence': 0.87,
    'method': 'face'  # or 'body' or 'unknown'
}
```

### Database Manager (`database_manager.py`)

**Purpose**: Log presence data with smart re-appearance handling

**Key Features**:
- 30-second threshold for re-appearances (configurable)
- Automatic session merging
- Duration calculation
- Active session tracking

**Standalone Usage**:
```python
from database_manager import DatabaseManager

db = DatabaseManager(db_path="presence_log.db", reappear_threshold=30)
log_id = db.log_detection("John", track_id=1)
```

**Re-appearance Logic**:
1. If same person reappears within threshold → Update existing entry's `end_time`
2. If same person reappears after threshold → Create new entry
3. If person leaves for >60 seconds → Finalize entry (calculate duration, set status='completed')

### Integrated System (`integrated_system.py`)

**Purpose**: Main pipeline connecting all modules

**Workflow**:
1. Read frame from camera
2. Track objects using YOLO11
3. For each tracked person:
   - Extract cropped image
   - Try face recognition
   - If face fails, try body embedding matching
   - Log to database with track ID
4. Periodically cleanup inactive tracks
5. Display annotated video with UI

**Controls**:
- `q` - Quit
- `s` - Show statistics
- `r` - Reset statistics

## 🎯 How the 30-Second Threshold Works

### Scenario 1: Person stays in frame
```
Person enters → Track ID: 1 → Face recognized: "John"
Log: id=1, user=John, start=10:00:00, end=10:00:00, status=active

[5 seconds later, still in frame]
Update: id=1, user=John, start=10:00:00, end=10:00:05, status=active

[20 seconds later, still in frame]
Update: id=1, user=John, start=10:00:00, end=10:00:25, status=active
```

### Scenario 2: Person leaves and returns within 30 seconds
```
Person enters → Track ID: 1 → Recognized: "John"
Log: id=1, user=John, start=10:00:00, end=10:00:10, status=active

[Person leaves frame at 10:00:10]

[Person re-enters at 10:00:25 (15 seconds later) → Track ID: 3]
SAME ENTRY: id=1, user=John, start=10:00:00, end=10:00:25, status=active
(Entry is reused, not created new)
```

### Scenario 3: Person leaves and returns after 30 seconds
```
Person enters → Track ID: 1 → Recognized: "John"
Log: id=1, user=John, start=10:00:00, end=10:00:10, status=active

[Person leaves at 10:00:10]
[After 60 seconds of inactivity, entry is finalized]
Update: id=1, user=John, start=10:00:00, end=10:00:10, duration=10, status=completed

[Person re-enters at 10:01:00 (50 seconds later) → Track ID: 5]
NEW ENTRY: id=2, user=John, start=10:01:00, end=10:01:00, status=active
```

## 📈 System Workflow

```
┌─────────────┐
│   Camera    │
└──────┬──────┘
       │ frame
       ▼
┌─────────────────────┐
│  Object Tracker     │
│  (YOLO11 + Track)   │
└──────┬──────────────┘
       │ detections (track_id, bbox, cropped_img)
       ▼
┌──────────────────────────────────┐
│  Face Recognition Module         │
│  1. Try face recognition         │
│  2. Fallback to body embedding   │
└──────┬───────────────────────────┘
       │ (identity, confidence, method)
       ▼
┌──────────────────────────────────┐
│  Database Manager                │
│  1. Check for active entry       │
│  2. Check re-appearance time     │
│  3. Update or create entry       │
└──────────────────────────────────┘
```

## 🎛️ Configuration

### Thresholds

Edit in `integrated_system.py`:
```python
reappear_threshold = 30  # Seconds to consider as same session
cleanup_interval = 60    # Seconds before finalizing inactive sessions
```

### Face Recognition

Edit in `face_recognition_module.py`:
```python
face_threshold = 0.5      # Confidence threshold for face recognition
body_threshold = 0.7      # Similarity threshold for body matching
detector_backend = "opencv"  # or "retinaface", "mtcnn", "ssd", "dlib"
```

### Object Tracking

Edit in `object_tracker.py`:
```python
conf_threshold = 0.5  # Detection confidence threshold
```

## 📝 File Structure

```
iot_project/
├── object_tracker.py           # Object detection and tracking
├── face_recognition_module.py  # Face recognition and body embeddings
├── database_manager.py         # Database operations and logging
├── integrated_system.py        # Main pipeline
├── datacollect.py             # Data collection tool
├── train_face_recog.py        # Model training script
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── best.pt                    # YOLO model (download from Ultralytics)
├── keras_model.h5            # Trained face model (generated)
├── labels.txt                # Class labels (generated)
├── presence_log.db           # SQLite database (generated)
└── images/                   # Training data directory
    ├── person1/
    ├── person2/
    └── person3/
```

## 🔍 Querying the Database

### Using Python

```python
from database_manager import DatabaseManager

db = DatabaseManager()

# Get active sessions
active = db.get_active_sessions()

# Get today's sessions
today = db.get_sessions_by_date()

# Get total time for a user
total_seconds = db.get_user_total_time("John")
```

### Using SQL

```cmd
sqlite3 presence_log.db

-- View all sessions today
SELECT * FROM presence_log WHERE date = date('now');

-- Get total time per user today
SELECT user, SUM(duration) as total_seconds 
FROM presence_log 
WHERE date = date('now') AND duration IS NOT NULL
GROUP BY user;

-- View active sessions
SELECT * FROM presence_log WHERE status = 'active';
```

## ⚙️ Advanced Usage

### Running Individual Modules

**Test Object Tracker Only**:
```cmd
python object_tracker.py
```

**Test Face Recognition Only**:
```cmd
python face_recognition_module.py
```

**Test Database Manager**:
```cmd
python database_manager.py
```

### Custom Integration

```python
from object_tracker import ObjectTracker
from face_recognition_module import FaceRecognitionModule
from database_manager import DatabaseManager

# Initialize modules
tracker = ObjectTracker()
recognizer = FaceRecognitionModule()
db = DatabaseManager()

# Process frame
detections = tracker.process_frame(frame)

for det in detections:
    # Recognize person
    result = recognizer.process_detection(det['cropped_img'], det['track_id'])
    
    # Log if recognized
    if result['identity']:
        db.log_detection(result['identity'], det['track_id'])
```

## 🐛 Troubleshooting

### "Model file not found"
- Make sure you've downloaded or trained the models
- For YOLO: Download from Ultralytics or use your custom trained model
- For Face Recognition: Run `train_face_recog.py` after collecting data

### "No faces detected"
- Try different `detector_backend` in `face_recognition_module.py`
- Options: `opencv` (fastest), `retinaface` (best), `mtcnn`, `ssd`, `dlib`
- Ensure good lighting conditions

### "Low FPS"
- Use a smaller YOLO model (yolo11n.pt is fastest)
- Reduce camera resolution in `integrated_system.py`
- Use GPU if available (install CUDA-enabled TensorFlow)

### "Database locked"
- The system uses threading locks to prevent concurrent access
- If you're accessing the database externally, close other connections first

## 📚 References

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV](https://docs.opencv.org/)

## 📄 License

MIT License - Feel free to use and modify for your projects!

## 🤝 Contributing

This is a modular system designed for easy extension. Feel free to:
- Add new recognition methods
- Improve body embedding algorithms
- Add new database backends
- Enhance the UI

---

**Author**: GitHub Copilot  
**Date**: November 2025
