"""
Simplified Database Manager
Simple logging of entry and exit times for each detection.
No threshold logic - every detection is logged separately.
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict
import threading


class DatabaseManager:
    """
    Manages SQLite database for person presence logging.
    Simple entry/exit logging without complex threshold logic.
    """
    
    def __init__(self, db_path: str = "presence_log.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Simple presence log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS presence_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    duration INTEGER,
                    date TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_identity 
                ON presence_log(identity)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_date 
                ON presence_log(date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_track_id 
                ON presence_log(track_id)
            """)
            
            conn.commit()
            conn.close()
            
        print(f"Database initialized at {self.db_path}")
    
    def _get_current_datetime(self) -> tuple:
        """
        Get current date and datetime as strings.
        
        Returns:
            Tuple of (date_str, datetime_str)
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
        return date_str, datetime_str
    
    def log_entry(self, identity: str, track_id: int) -> int:
        """
        Log a person entry (detection started).
        
        Args:
            identity: Person's identity (name or "Guest")
            track_id: Track ID from object tracker
            
        Returns:
            Entry ID (primary key)
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            date_str, datetime_str = self._get_current_datetime()
            
            cursor.execute("""
                INSERT INTO presence_log (identity, track_id, entry_time, date)
                VALUES (?, ?, ?, ?)
            """, (identity, track_id, datetime_str, date_str))
            
            entry_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return entry_id
    
    def log_exit(self, entry_id: int):
        """
        Log a person exit (track lost).
        
        Args:
            entry_id: Entry ID from log_entry
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            _, exit_datetime_str = self._get_current_datetime()
            
            # Get entry time to calculate duration
            cursor.execute("""
                SELECT entry_time
                FROM presence_log
                WHERE id = ?
            """, (entry_id,))
            
            result = cursor.fetchone()
            
            if result:
                entry_time_str = result[0]
                
                # Calculate duration in seconds
                entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                exit_time = datetime.strptime(exit_datetime_str, "%Y-%m-%d %H:%M:%S")
                duration = int((exit_time - entry_time).total_seconds())
                
                # Update exit time and duration
                cursor.execute("""
                    UPDATE presence_log
                    SET exit_time = ?, duration = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (exit_datetime_str, duration, entry_id))
                
                conn.commit()
            
            conn.close()
    
    def update_identity(self, entry_id: int, new_identity: str):
        """
        Update the identity for an entry.
        Useful when a "Guest" is later recognized.
        
        Args:
            entry_id: Entry ID to update
            new_identity: New identity name
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE presence_log
                SET identity = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_identity, entry_id))
            
            conn.commit()
            conn.close()
    
    def get_all_entries(self, limit: int = 100) -> List[Dict]:
        """
        Get recent entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of entry dictionaries
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, identity, track_id, entry_time, exit_time, duration, date
                FROM presence_log
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            entries = []
            for row in rows:
                entries.append({
                    'id': row[0],
                    'identity': row[1],
                    'track_id': row[2],
                    'entry_time': row[3],
                    'exit_time': row[4],
                    'duration': row[5],
                    'date': row[6]
                })
            
            return entries
    
    def get_entries_by_date(self, date: Optional[str] = None) -> List[Dict]:
        """
        Get all entries for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format (defaults to today)
            
        Returns:
            List of entry dictionaries
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, identity, track_id, entry_time, exit_time, duration, date
                FROM presence_log
                WHERE date = ?
                ORDER BY entry_time DESC
            """, (date,))
            
            rows = cursor.fetchall()
            conn.close()
            
            entries = []
            for row in rows:
                entries.append({
                    'id': row[0],
                    'identity': row[1],
                    'track_id': row[2],
                    'entry_time': row[3],
                    'exit_time': row[4],
                    'duration': row[5],
                    'date': row[6]
                })
            
            return entries
    
    def get_entries_by_identity(self, identity: str, date: Optional[str] = None) -> List[Dict]:
        """
        Get all entries for a specific identity.
        
        Args:
            identity: Person's identity
            date: Optional date filter in YYYY-MM-DD format
            
        Returns:
            List of entry dictionaries
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if date:
                cursor.execute("""
                    SELECT id, identity, track_id, entry_time, exit_time, duration, date
                    FROM presence_log
                    WHERE identity = ? AND date = ?
                    ORDER BY entry_time DESC
                """, (identity, date))
            else:
                cursor.execute("""
                    SELECT id, identity, track_id, entry_time, exit_time, duration, date
                    FROM presence_log
                    WHERE identity = ?
                    ORDER BY entry_time DESC
                """, (identity,))
            
            rows = cursor.fetchall()
            conn.close()
            
            entries = []
            for row in rows:
                entries.append({
                    'id': row[0],
                    'identity': row[1],
                    'track_id': row[2],
                    'entry_time': row[3],
                    'exit_time': row[4],
                    'duration': row[5],
                    'date': row[6]
                })
            
            return entries
    
    def get_total_time(self, identity: str, date: Optional[str] = None) -> int:
        """
        Get total time (in seconds) for a person.
        
        Args:
            identity: Person's identity
            date: Optional date filter in YYYY-MM-DD format (defaults to today)
            
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
                WHERE identity = ? AND date = ? AND duration IS NOT NULL
            """, (identity, date))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result[0] else 0
    
    def get_active_entries(self) -> List[Dict]:
        """
        Get all entries that don't have an exit time yet.
        
        Returns:
            List of active entry dictionaries
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, identity, track_id, entry_time, date
                FROM presence_log
                WHERE exit_time IS NULL
                ORDER BY entry_time DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            entries = []
            for row in rows:
                entries.append({
                    'id': row[0],
                    'identity': row[1],
                    'track_id': row[2],
                    'entry_time': row[3],
                    'date': row[4]
                })
            
            return entries


def main():
    """
    Demo/test the simplified database manager.
    """
    import time
    
    db = DatabaseManager(db_path="test_presence_simple.db")
    
    print("Simplified Database Manager Demo")
    print("-" * 60)
    
    # Simulate some detections
    print("\n1. Logging entry for Alice (track_id=1)")
    entry_id_1 = db.log_entry("Alice", track_id=1)
    print(f"   Entry ID: {entry_id_1}")
    
    time.sleep(2)
    
    print("\n2. Logging entry for Bob (track_id=2)")
    entry_id_2 = db.log_entry("Bob", track_id=2)
    print(f"   Entry ID: {entry_id_2}")
    
    time.sleep(2)
    
    print("\n3. Logging entry for Guest (track_id=3)")
    entry_id_3 = db.log_entry("Guest", track_id=3)
    print(f"   Entry ID: {entry_id_3}")
    
    time.sleep(1)
    
    print("\n4. Updating Guest to Charlie")
    db.update_identity(entry_id_3, "Charlie")
    print("   Identity updated")
    
    time.sleep(2)
    
    print("\n5. Logging exits")
    db.log_exit(entry_id_1)
    db.log_exit(entry_id_2)
    db.log_exit(entry_id_3)
    print("   All exits logged")
    
    print("\n6. Today's entries:")
    entries = db.get_entries_by_date()
    for entry in entries:
        print(f"   {entry}")
    
    print("\n7. Total time for Alice today:")
    total = db.get_total_time("Alice")
    print(f"   {total} seconds")
    
    print("\n8. Active entries:")
    active = db.get_active_entries()
    if active:
        for entry in active:
            print(f"   {entry}")
    else:
        print("   No active entries")


if __name__ == "__main__":
    main()
