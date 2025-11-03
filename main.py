import base64
import io
import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from PIL import Image
import numpy as np

# local helper (below)
from face_recognizer import FaceRecognizer, ensure_db

# Config
DB_PATH = "people_presence.db"
REAPPEAR_SECONDS = 8  # x seconds tolerance
FACE_MATCH_THRESHOLD = 0.5

app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")

# Globals
recognizer = FaceRecognizer(db_path=DB_PATH, face_threshold=FACE_MATCH_THRESHOLD)
db_lock = threading.Lock()

# Ensure DB/tables exist
ensure_db(DB_PATH)


def now_ts():
    return datetime.now(timezone.utc).isoformat()


def save_raw_detection(conn, payload, face_detected, face_confidence, recognized_person_id):
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO raw_detections(timestamp, tracker_id, face_detected, face_confidence, recognized_person_id)
           VALUES (?, ?, ?, ?, ?)""",
        (payload["timestamp"], str(payload["tracker_id"]), int(face_detected), face_confidence, recognized_person_id),
    )
    conn.commit()


def find_active_presence_by_tracker(conn, tracker_id):
    cur = conn.cursor()
    cur.execute(
        "SELECT log_id, person_id, start_time, last_seen_time FROM presence_log WHERE tracker_id=? AND status='active' LIMIT 1",
        (tracker_id,),
    )
    return cur.fetchone()


def find_recent_presence_by_person(conn, person_id, within_seconds):
    cur = conn.cursor()
    cutoff = datetime.now(timezone.utc).timestamp() - within_seconds
    cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()
    cur.execute(
        "SELECT log_id, tracker_id, last_seen_time FROM presence_log WHERE person_id=? AND status='active' AND last_seen_time>=? LIMIT 1",
        (person_id, cutoff_iso),
    )
    return cur.fetchone()


def create_presence(conn, person_id, tracker_id, ts):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO presence_log(person_id, tracker_id, start_time, last_seen_time, accumulated_seconds, status) VALUES (?, ?, ?, ?, ?, ?)",
        (person_id, tracker_id, ts, ts, 0, "active"),
    )
    conn.commit()
    return cur.lastrowid


def update_presence_last_seen(conn, log_id, ts):
    cur = conn.cursor()
    cur.execute(
        "UPDATE presence_log SET last_seen_time=? WHERE log_id=?",
        (ts, log_id),
    )
    conn.commit()


def end_presence(conn, log_id):
    cur = conn.cursor()
    cur.execute("SELECT start_time, last_seen_time FROM presence_log WHERE log_id=?", (log_id,))
    row = cur.fetchone()
    if not row:
        return
    start_time_ts = datetime.fromisoformat(row[0]).timestamp()
    last_seen_ts = datetime.fromisoformat(row[1]).timestamp()
    accumulated = int(last_seen_ts - start_time_ts)
    end_time_iso = datetime.fromtimestamp(last_seen_ts, timezone.utc).isoformat()
    cur.execute(
        "UPDATE presence_log SET end_time=?, accumulated_seconds=?, status='ended' WHERE log_id=?",
        (end_time_iso, accumulated, log_id),
    )
    conn.commit()


@app.route("/api/detection", methods=["POST"])
def receive_detection():
    """
    Expected JSON:
    {
      "tracker_id": "123",
      "timestamp": "ISO string",
      "image_b64": "<base64 of cropped person image (RGB)>"
      "bbox": [x,y,w,h]  # optional
    }
    """
    try:
        payload = request.get_json()
        if payload is None:
            return jsonify({"error": "invalid json"}), 400
        tracker_id = str(payload.get("tracker_id"))
        ts = payload.get("timestamp", now_ts())
        img_b64 = payload.get("image_b64")
        if not tracker_id or not img_b64:
            return jsonify({"error": "missing fields"}), 400

        # decode image
        try:
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)
        except Exception as e:
            return jsonify({"error": "invalid image", "detail": str(e)}), 400

        # run face recognition
        try:
            face_found, face_conf, matched_person_id = recognizer.recognize(img_np)
        except Exception as e:
            face_found, face_conf, matched_person_id = False, None, None

        # DB operations protected by lock
        with db_lock:
            conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            save_raw_detection(conn, {"timestamp": ts, "tracker_id": tracker_id}, face_found, face_conf or None, matched_person_id)
            active = find_active_presence_by_tracker(conn, tracker_id)
            if matched_person_id:
                # if tracker already has active presence but different person_id, close old or reassign
                if active:
                    if active[1] != matched_person_id:
                        # end old presence
                        end_presence(conn, active[0])
                        # create new with recognized person
                        log_id = create_presence(conn, matched_person_id, tracker_id, ts)
                    else:
                        update_presence_last_seen(conn, active[0], ts)
                        log_id = active[0]
                else:
                    # try match recent presence for same person
                    recent = find_recent_presence_by_person(conn, matched_person_id, REAPPEAR_SECONDS)
                    if recent:
                        # reuse
                        update_presence_last_seen(conn, recent[0], ts)
                        log_id = recent[0]
                    else:
                        log_id = create_presence(conn, matched_person_id, tracker_id, ts)
            else:
                # no face recognized
                if active:
                    update_presence_last_seen(conn, active[0], ts)
                    log_id = active[0]
                else:
                    # try to find any active presence with same appearance? (optional)
                    log_id = create_presence(conn, None, tracker_id, ts)

            # fetch current presence row
            cur = conn.cursor()
            cur.execute("SELECT log_id, person_id, tracker_id, start_time, last_seen_time FROM presence_log WHERE log_id=?", (log_id,))
            presence_row = cur.fetchone()
            conn.close()

        # emit websocket update
        info = {
            "log_id": presence_row[0],
            "person_id": presence_row[1],
            "tracker_id": presence_row[2],
            "start_time": presence_row[3],
            "last_seen_time": presence_row[4],
            "face_found": bool(face_found),
            "face_confidence": face_conf,
            "server_timestamp": now_ts(),
        }
        socketio.emit("presence_update", info)

        return jsonify({"status": "ok", "assigned_log_id": log_id, "recognized_person_id": matched_person_id}), 200

    except Exception as exc:
        return jsonify({"error": "server error", "detail": str(exc)}), 500


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# background thread to sweep ended presences
def sweeper():
    while True:
        try:
            time.sleep(1.0)
            with db_lock:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cutoff = datetime.now(timezone.utc).timestamp() - REAPPEAR_SECONDS
                cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()
                # find active presences whose last_seen_time < cutoff
                cur.execute("SELECT log_id FROM presence_log WHERE status='active' AND last_seen_time<?", (cutoff_iso,))
                rows = cur.fetchall()
                for (log_id,) in rows:
                    end_presence(conn, log_id)
                    # emit ended event
                    socketio.emit("presence_ended", {"log_id": log_id})
                conn.close()
        except Exception as e:
            print("Sweeper error:", e)


if __name__ == "__main__":
    t = threading.Thread(target=sweeper, daemon=True)
    t.start()
    socketio.run(app, host="0.0.0.0", port=5000)
