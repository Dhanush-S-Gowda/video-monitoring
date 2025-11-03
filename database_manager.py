"""
Database Manager Module
Handles SQLite database operations for logging person presence.
Implements 30-second threshold logic for handling re-appearances.
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import threading


class DatabaseManager:
    """
    Manages SQLite database for person presence logging.
    """
    
    def __init__(self, db_path: str = "presence_log.db", reappear_threshold: int = 30):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
            reappear_threshold: Threshold in seconds for considering re-appearances
        """
        self.db_path = db_path
        self.reappear_threshold = reappear_threshold
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main presence log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS presence_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    user TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration INTEGER,
                    track_id INTEGER,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_status 
                ON presence_log(user, status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_date 
                ON presence_log(date)
            """)
            
            # Tracking information table (maps track_id to user)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS track_mapping (
                    track_id INTEGER PRIMARY KEY,
                    user TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    presence_log_id INTEGER,
                    FOREIGN KEY (presence_log_id) REFERENCES presence_log(id)
                )
            """)
            
            conn.commit()
            conn.close()
            
        print(f"Database initialized at {self.db_path}")
    
    def _get_current_datetime(self) -> Tuple[str, str]:
        """
        Get current date and time as strings.
        
        Returns:
            Tuple of (date_str, time_str)
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        return date_str, time_str
    
    def _get_datetime_obj(self, date_str: str, time_str: str) -> datetime:
        """
        Convert date and time strings to datetime object.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            time_str: Time string in HH:MM:SS format
            
        Returns:
            datetime object
        """
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    
    def log_detection(self, user: str, track_id: int) -> int:
        """
        Log a person detection. Handles re-appearances within threshold.
        
        Args:
            user: User name/identity
            track_id: Track ID from object tracker
            
        Returns:
            presence_log_id of the created or updated entry
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            date_str, time_str = self._get_current_datetime()
            current_time = self._get_datetime_obj(date_str, time_str)
            
            # Check if track_id already has an active mapping
            cursor.execute("""
                SELECT user, last_seen, presence_log_id
                FROM track_mapping
                WHERE track_id = ?
            """, (track_id,))
            
            track_mapping = cursor.fetchone()
            
            if track_mapping:
                # Track exists, update last_seen
                mapped_user, last_seen_str, presence_log_id = track_mapping
                
                cursor.execute("""
                    UPDATE track_mapping
                    SET last_seen = ?, user = ?
                    WHERE track_id = ?
                """, (time_str, user, track_id))
                
                # Update the presence log entry
                cursor.execute("""
                    UPDATE presence_log
                    SET end_time = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (time_str, presence_log_id))
                
                conn.commit()
                conn.close()
                return presence_log_id
            
            # Check for recent entry for this user (within threshold)
            threshold_time = current_time - timedelta(seconds=self.reappear_threshold)
            threshold_time_str = threshold_time.strftime("%H:%M:%S")
            
            cursor.execute("""
                SELECT id, start_time, end_time
                FROM presence_log
                WHERE user = ? AND date = ? AND status = 'active'
                AND end_time >= ?
                ORDER BY id DESC
                LIMIT 1
            """, (user, date_str, threshold_time_str))
            
            recent_entry = cursor.fetchone()
            
            if recent_entry:
                # Update existing entry (same user re-appeared within threshold)
                presence_log_id, start_time, end_time = recent_entry
                
                cursor.execute("""
                    UPDATE presence_log
                    SET end_time = ?, track_id = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (time_str, track_id, presence_log_id))
                
                # Update track mapping
                cursor.execute("""
                    INSERT OR REPLACE INTO track_mapping (track_id, user, last_seen, presence_log_id)
                    VALUES (?, ?, ?, ?)
                """, (track_id, user, time_str, presence_log_id))
                
                conn.commit()
                conn.close()
                return presence_log_id
            
            else:
                # Create new entry (new appearance or beyond threshold)
                cursor.execute("""
                    INSERT INTO presence_log (date, user, start_time, end_time, track_id, status)
                    VALUES (?, ?, ?, ?, ?, 'active')
                """, (date_str, user, time_str, time_str, track_id))
                
                presence_log_id = cursor.lastrowid
                
                # Create track mapping
                cursor.execute("""
                    INSERT OR REPLACE INTO track_mapping (track_id, user, last_seen, presence_log_id)
                    VALUES (?, ?, ?, ?)
                """, (track_id, user, time_str, presence_log_id))
                
                conn.commit()
                conn.close()
                return presence_log_id
    
    def finalize_entry(self, presence_log_id: int):
        """
        Finalize a presence log entry by calculating duration and setting status.
        
        Args:
            presence_log_id: ID of the presence log entry
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT date, start_time, end_time
                FROM presence_log
                WHERE id = ?
            """, (presence_log_id,))
            
            entry = cursor.fetchone()
            
            if entry:
                date_str, start_time_str, end_time_str = entry
                
                start_dt = self._get_datetime_obj(date_str, start_time_str)
                end_dt = self._get_datetime_obj(date_str, end_time_str)
                
                duration = int((end_dt - start_dt).total_seconds())
                
                cursor.execute("""
                    UPDATE presence_log
                    SET duration = ?, status = 'completed', updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (duration, presence_log_id))
                
                conn.commit()
            
            conn.close()
    
    def cleanup_inactive_tracks(self, timeout: int = 60):
        """
        Clean up track mappings that haven't been seen recently and finalize their entries.
        
        Args:
            timeout: Time in seconds to consider a track inactive
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            date_str, current_time_str = self._get_current_datetime()
            current_dt = self._get_datetime_obj(date_str, current_time_str)
            cutoff_dt = current_dt - timedelta(seconds=timeout)
            cutoff_time_str = cutoff_dt.strftime("%H:%M:%S")
            
            # Find inactive tracks
            cursor.execute("""
                SELECT track_id, presence_log_id
                FROM track_mapping
                WHERE last_seen < ?
            """, (cutoff_time_str,))
            
            inactive_tracks = cursor.fetchall()
            
            for track_id, presence_log_id in inactive_tracks:
                # Finalize the presence log entry
                self.finalize_entry(presence_log_id)
                
                # Remove track mapping
                cursor.execute("""
                    DELETE FROM track_mapping
                    WHERE track_id = ?
                """, (track_id,))
            
            conn.commit()
            conn.close()
            
            if inactive_tracks:
                print(f"Cleaned up {len(inactive_tracks)} inactive tracks")
    
    def get_active_sessions(self) -> List[Dict]:
        """
        Get all currently active presence sessions.
        
        Returns:
            List of active session dictionaries
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, date, user, start_time, end_time, track_id
                FROM presence_log
                WHERE status = 'active'
                ORDER BY id DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            sessions = []
            for row in rows:
                sessions.append({
                    'id': row[0],
                    'date': row[1],
                    'user': row[2],
                    'start_time': row[3],
                    'end_time': row[4],
                    'track_id': row[5]
                })
            
            return sessions
    
    def get_sessions_by_date(self, date: Optional[str] = None) -> List[Dict]:
        """
        Get all sessions for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format (defaults to today)
            
        Returns:
            List of session dictionaries
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, date, user, start_time, end_time, duration, status
                FROM presence_log
                WHERE date = ?
                ORDER BY start_time DESC
            """, (date,))
            
            rows = cursor.fetchall()
            conn.close()
            
            sessions = []
            for row in rows:
                sessions.append({
                    'id': row[0],
                    'date': row[1],
                    'user': row[2],
                    'start_time': row[3],
                    'end_time': row[4],
                    'duration': row[5],
                    'status': row[6]
                })
            
            return sessions
    
    def get_user_total_time(self, user: str, date: Optional[str] = None) -> int:
        """
        Get total time (in seconds) for a user on a specific date.
        
        Args:
            user: User name
            date: Date string in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Total duration in seconds
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT SUM(duration)
                FROM presence_log
                WHERE user = ? AND date = ? AND duration IS NOT NULL
            """, (user, date))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result[0] else 0


def main():
    """
    Demo/test the database manager.
    """
    db = DatabaseManager(db_path="test_presence.db", reappear_threshold=30)
    
    print("Database Manager Demo")
    print("-" * 50)
    
    # Simulate some detections
    print("\n1. Logging first detection for Alice (track_id=1)")
    log_id = db.log_detection("Alice", track_id=1)
    print(f"   Created log entry ID: {log_id}")
    
    import time
    time.sleep(2)
    
    print("\n2. Logging continued detection for Alice (track_id=1)")
    log_id = db.log_detection("Alice", track_id=1)
    print(f"   Updated log entry ID: {log_id}")
    
    print("\n3. Logging detection for Bob (track_id=2)")
    log_id2 = db.log_detection("Bob", track_id=2)
    print(f"   Created log entry ID: {log_id2}")
    
    time.sleep(2)
    
    print("\n4. Simulating Alice leaving and returning within 30 seconds")
    time.sleep(2)
    log_id = db.log_detection("Alice", track_id=3)
    print(f"   Should reuse log entry ID: {log_id}")
    
    print("\n5. Active sessions:")
    sessions = db.get_active_sessions()
    for session in sessions:
        print(f"   {session}")
    
    print("\n6. Finalizing entries")
    db.finalize_entry(log_id)
    db.finalize_entry(log_id2)
    
    print("\n7. Today's sessions:")
    today_sessions = db.get_sessions_by_date()
    for session in today_sessions:
        print(f"   {session}")
    
    print("\n8. Total time for Alice today:")
    total = db.get_user_total_time("Alice")
    print(f"   {total} seconds")


if __name__ == "__main__":
    main()
