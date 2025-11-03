# Quick Reference Card

## 🚀 Quick Start Commands

```bash
# Setup
pip install -r requirements.txt

# Collect data for person 1
python datacollect.py

# Collect data for person 2  
python datacollect.py

# Train the model
python train_face_recog.py

# Run the system
python integrated_system.py
```

## 📁 File Reference

| File | Purpose | Run Standalone? |
|------|---------|-----------------|
| `integrated_system.py` | Main system | ✅ Yes - **START HERE** |
| `object_tracker.py` | Object tracking only | ✅ Yes |
| `face_recognition_module.py` | Face recognition only | ✅ Yes |
| `database_manager.py` | Database operations | ✅ Yes |
| `datacollect.py` | Collect training images | ✅ Yes |
| `train_face_recog.py` | Train face model | ✅ Yes |
| `view_database.py` | View database | ✅ Yes |
| `start_guide.py` | Interactive guide | ✅ Yes |
| `setup.bat` | Windows setup | ✅ Yes (Windows) |

## ⌨️ Keyboard Controls (Integrated System)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Show statistics |
| `r` | Reset statistics |

## 🗂️ Database Queries

### View Database Tool
```bash
python view_database.py --today          # Today's sessions
python view_database.py --active         # Active sessions
python view_database.py --stats          # User statistics
python view_database.py --user Alice     # Alice's sessions
python view_database.py --all            # All sessions
```

### Direct SQL
```bash
sqlite3 presence_log.db
sqlite> SELECT * FROM presence_log WHERE date = date('now');
sqlite> SELECT user, SUM(duration) FROM presence_log GROUP BY user;
```

## 🎯 Module Usage

### Object Tracker
```python
from object_tracker import ObjectTracker

tracker = ObjectTracker(model_path="best.pt")
detections = tracker.process_frame(frame)
# Returns: [{'track_id': 1, 'bbox': [...], 'cropped_img': ...}]
```

### Face Recognition
```python
from face_recognition_module import FaceRecognitionModule

recognizer = FaceRecognitionModule()
result = recognizer.process_detection(cropped_img, track_id)
# Returns: {'identity': 'Alice', 'confidence': 0.87, 'method': 'face'}
```

### Database Manager
```python
from database_manager import DatabaseManager

db = DatabaseManager(reappear_threshold=30)
log_id = db.log_detection("Alice", track_id=1)
sessions = db.get_active_sessions()
```

## 📊 Database Schema

### presence_log
```sql
id              INTEGER PRIMARY KEY
date            TEXT (YYYY-MM-DD)
user            TEXT (person name)
start_time      TEXT (HH:MM:SS)
end_time        TEXT (HH:MM:SS)
duration        INTEGER (seconds)
track_id        INTEGER
status          TEXT ('active' or 'completed')
created_at      TEXT (timestamp)
updated_at      TEXT (timestamp)
```

## ⚙️ Configuration

### Command Line
```bash
python integrated_system.py \
    --yolo-model best.pt \
    --face-model keras_model.h5 \
    --threshold 30 \
    --camera 0
```

### In Code

**Object Tracker**
```python
ObjectTracker(
    model_path="best.pt",
    conf_threshold=0.5
)
```

**Face Recognition**
```python
FaceRecognitionModule(
    model_path="keras_model.h5",
    labels_path="labels.txt",
    face_threshold=0.5,
    body_threshold=0.7,
    detector_backend="opencv"  # opencv, retinaface, mtcnn
)
```

**Database**
```python
DatabaseManager(
    db_path="presence_log.db",
    reappear_threshold=30  # seconds
)
```

## 🔄 Re-Appearance Logic

| Condition | Action | Result |
|-----------|--------|--------|
| Person in frame | Update end_time | Single entry |
| Leave < 30s | Reuse entry | Single entry |
| Leave > 30s | New entry | Multiple entries |
| Inactive > 60s | Finalize | Status = completed |

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| No camera | Check camera index (--camera 0, 1, 2...) |
| Low FPS | Use yolo11n.pt, reduce resolution |
| No faces detected | Try detector_backend="retinaface" |
| Model not found | Run train_face_recog.py first |
| Import errors | pip install -r requirements.txt |

## 📦 Required Files

### Before Running
- ✅ requirements.txt
- ✅ Python 3.8+

### After Data Collection
- ✅ images/person1/*.jpg (300+ images)
- ✅ images/person2/*.jpg (300+ images)

### After Training
- ✅ keras_model.h5
- ✅ labels.txt

### For Tracking
- ✅ best.pt or yolo11n.pt (auto-downloads)

## 🎓 Learning Path

1. **Start**: Run `python start_guide.py`
2. **Test Tracker**: `python object_tracker.py`
3. **Test Face Recognition**: `python face_recognition_module.py`
4. **Collect Data**: `python datacollect.py` (repeat for each person)
5. **Train Model**: `python train_face_recog.py`
6. **Run System**: `python integrated_system.py`
7. **View Data**: `python view_database.py --today`

## 💡 Tips

- Collect 300+ images per person for best accuracy
- Use good lighting for data collection
- Vary angles and expressions during collection
- Start with yolo11n.pt (fastest)
- Use GPU for better performance
- Check logs with `python view_database.py`
- Test modules individually before integration

## 📞 Common Use Cases

### Attendance System
```bash
python integrated_system.py --threshold 60
# Then query: 
python view_database.py --stats
```

### Security Monitoring
```bash
python integrated_system.py --threshold 30
# Real-time display with track IDs
```

### Time Tracking
```bash
# After running system:
python view_database.py --user Alice --days 7
# Shows Alice's presence for last 7 days
```

## 🔗 Dependencies

```
ultralytics      # YOLO11
opencv-python    # Computer vision
tensorflow       # CNN model
deepface         # Face detection/recognition
numpy            # Array operations
Pillow           # Image processing
```

## 📈 Performance

| Model | FPS | Accuracy | Use Case |
|-------|-----|----------|----------|
| yolo11n | 25-30 | Good | Real-time |
| yolo11s | 20-25 | Better | Balanced |
| yolo11m | 15-20 | Best | Accuracy |

---

**Quick Help**: Run `python start_guide.py` for interactive setup

**Documentation**: See `README.md` for complete guide

**Project Details**: See `PROJECT_SUMMARY.md` for architecture
