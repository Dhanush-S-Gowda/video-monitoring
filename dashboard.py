"""
Real-time Dashboard for Person Tracking System
Web-based interface to display database logs while main.py is running.
Efficiently handles concurrent database access.
"""

from flask import Flask, render_template, jsonify
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading
import time

app = Flask(__name__)

# Database path
DB_PATH = "presence_log.db"

# Cache for reducing database queries
cache_lock = threading.Lock()
cache_data = {
    'last_update': None,
    'data': None
}
CACHE_DURATION = 2  # seconds


def get_db_connection():
    """Get a database connection with read-only mode for safety."""
    conn = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def format_duration(seconds):
    """Format duration in seconds to human readable format."""
    if seconds is None:
        return "Active"
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_time(datetime_str):
    """Extract time from datetime string."""
    if not datetime_str:
        return None
    try:
        if ' ' in datetime_str:
            return datetime_str.split()[1]
        return datetime_str
    except:
        return datetime_str


def combine_close_entries(entries: List[Dict], threshold_minutes: int = 1) -> List[Dict]:
    """
    Combine entries where exit time of one is close to entry time of next.
    
    Args:
        entries: List of entry dictionaries sorted by entry_time
        threshold_minutes: Minutes threshold for combining entries
        
    Returns:
        List of combined entries
    """
    if not entries:
        return []
    
    combined = []
    current_group = [entries[0]]
    
    for i in range(1, len(entries)):
        prev_entry = current_group[-1]
        curr_entry = entries[i]
        
        # Skip if either entry is still active
        if not prev_entry['exit_time'] or not curr_entry['entry_time']:
            combined.append({
                'entry_time': current_group[0]['entry_time'],
                'exit_time': prev_entry['exit_time'],
                'duration': sum(e.get('duration', 0) or 0 for e in current_group),
                'entry_count': len(current_group)
            })
            current_group = [curr_entry]
            continue
        
        try:
            prev_exit = datetime.strptime(prev_entry['exit_time'], "%Y-%m-%d %H:%M:%S")
            curr_entry_time = datetime.strptime(curr_entry['entry_time'], "%Y-%m-%d %H:%M:%S")
            
            time_diff = (curr_entry_time - prev_exit).total_seconds() / 60
            
            if time_diff <= threshold_minutes:
                # Combine
                current_group.append(curr_entry)
            else:
                # Save current group and start new
                combined.append({
                    'entry_time': current_group[0]['entry_time'],
                    'exit_time': prev_entry['exit_time'],
                    'duration': sum(e.get('duration', 0) or 0 for e in current_group),
                    'entry_count': len(current_group)
                })
                current_group = [curr_entry]
        except:
            # If parsing fails, don't combine
            combined.append({
                'entry_time': current_group[0]['entry_time'],
                'exit_time': prev_entry['exit_time'],
                'duration': sum(e.get('duration', 0) or 0 for e in current_group),
                'entry_count': len(current_group)
            })
            current_group = [curr_entry]
    
    # Add last group
    if current_group:
        combined.append({
            'entry_time': current_group[0]['entry_time'],
            'exit_time': current_group[-1]['exit_time'],
            'duration': sum(e.get('duration', 0) or 0 for e in current_group),
            'entry_count': len(current_group)
        })
    
    return combined


def get_dashboard_data(date: Optional[str] = None):
    """Get all dashboard data efficiently."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Check cache
    with cache_lock:
        if cache_data['last_update'] and cache_data['data']:
            if (time.time() - cache_data['last_update']) < CACHE_DURATION:
                return cache_data['data']
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Total duration per person (today)
        cursor.execute("""
            SELECT 
                identity,
                COUNT(*) as visit_count,
                SUM(CASE WHEN duration IS NOT NULL THEN duration ELSE 0 END) as total_duration,
                SUM(CASE WHEN exit_time IS NULL THEN 1 ELSE 0 END) as currently_present
            FROM presence_log
            WHERE date = ?
            GROUP BY identity
            ORDER BY total_duration DESC
        """, (date,))
        
        person_stats = []
        for row in cursor.fetchall():
            person_stats.append({
                'identity': row['identity'],
                'visit_count': row['visit_count'],
                'total_duration': row['total_duration'],
                'total_duration_formatted': format_duration(row['total_duration']),
                'currently_present': row['currently_present'] > 0
            })
        
        # 2. Entry/Exit times per person (combined if close)
        cursor.execute("""
            SELECT identity
            FROM presence_log
            WHERE date = ?
            GROUP BY identity
        """, (date,))
        
        identities = [row['identity'] for row in cursor.fetchall()]
        
        person_timeline = {}
        for identity in identities:
            cursor.execute("""
                SELECT entry_time, exit_time, duration
                FROM presence_log
                WHERE identity = ? AND date = ?
                ORDER BY entry_time ASC
            """, (identity, date))
            
            entries = [dict(row) for row in cursor.fetchall()]
            combined = combine_close_entries(entries, threshold_minutes=1)
            
            person_timeline[identity] = combined
        
        # 3. Active sessions
        cursor.execute("""
            SELECT id, identity, track_id, entry_time
            FROM presence_log
            WHERE exit_time IS NULL
            ORDER BY entry_time DESC
        """)
        
        active_sessions = [dict(row) for row in cursor.fetchall()]
        
        # 4. Recent activity (last 20 entries)
        cursor.execute("""
            SELECT id, identity, track_id, entry_time, exit_time, duration
            FROM presence_log
            WHERE date = ?
            ORDER BY id DESC
            LIMIT 20
        """, (date,))
        
        recent_activity = [dict(row) for row in cursor.fetchall()]
        
        # 5. Insights
        cursor.execute("""
            SELECT COUNT(DISTINCT identity) as unique_visitors
            FROM presence_log
            WHERE date = ?
        """, (date,))
        unique_visitors = cursor.fetchone()['unique_visitors']
        
        cursor.execute("""
            SELECT COUNT(*) as total_entries
            FROM presence_log
            WHERE date = ?
        """, (date,))
        total_entries = cursor.fetchone()['total_entries']
        
        cursor.execute("""
            SELECT COUNT(*) as guest_count
            FROM presence_log
            WHERE date = ? AND identity = 'Guest'
        """, (date,))
        guest_count = cursor.fetchone()['guest_count']
        
        # Average visit duration
        cursor.execute("""
            SELECT AVG(duration) as avg_duration
            FROM presence_log
            WHERE date = ? AND duration IS NOT NULL
        """, (date,))
        avg_duration = cursor.fetchone()['avg_duration']
        
        # Peak hour (most entries)
        cursor.execute("""
            SELECT 
                CAST(strftime('%H', entry_time) AS INTEGER) as hour,
                COUNT(*) as entry_count
            FROM presence_log
            WHERE date = ?
            GROUP BY hour
            ORDER BY entry_count DESC
            LIMIT 1
        """, (date,))
        peak_hour_row = cursor.fetchone()
        peak_hour = peak_hour_row['hour'] if peak_hour_row else None
        
        conn.close()
        
        data = {
            'date': date,
            'person_stats': person_stats,
            'person_timeline': person_timeline,
            'active_sessions': active_sessions,
            'recent_activity': recent_activity,
            'insights': {
                'unique_visitors': unique_visitors,
                'total_entries': total_entries,
                'guest_count': guest_count,
                'avg_duration': format_duration(int(avg_duration)) if avg_duration else "0s",
                'peak_hour': f"{peak_hour}:00" if peak_hour is not None else "N/A",
                'active_count': len(active_sessions)
            }
        }
        
        # Update cache
        with cache_lock:
            cache_data['last_update'] = time.time()
            cache_data['data'] = data
        
        return data
        
    except sqlite3.OperationalError as e:
        # Database might be locked or not exist yet
        return {
            'error': f'Database error: {e}',
            'date': date,
            'person_stats': [],
            'person_timeline': {},
            'active_sessions': [],
            'recent_activity': [],
            'insights': {}
        }
    except Exception as e:
        return {
            'error': f'Error: {e}',
            'date': date,
            'person_stats': [],
            'person_timeline': {},
            'active_sessions': [],
            'recent_activity': [],
            'insights': {}
        }


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/data')
def api_data():
    """API endpoint for dashboard data."""
    date = datetime.now().strftime("%Y-%m-%d")
    data = get_dashboard_data(date)
    return jsonify(data)


@app.route('/api/data/<date>')
def api_data_date(date):
    """API endpoint for specific date."""
    data = get_dashboard_data(date)
    return jsonify(data)


def run_dashboard(host='127.0.0.1', port=5000, debug=False):
    """Run the dashboard server."""
    print(f"\n{'='*60}")
    print("Person Tracking Dashboard")
    print(f"{'='*60}")
    print(f"Dashboard URL: http://{host}:{port}")
    print(f"Database: {DB_PATH}")
    print("Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Person Tracking Dashboard")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--db', default='presence_log.db', help='Database path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    DB_PATH = args.db
    
    run_dashboard(host=args.host, port=args.port, debug=args.debug)
