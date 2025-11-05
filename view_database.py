"""
View Database - Simplified Version
Display entries from the simplified presence_log database.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional
import argparse


def format_duration(seconds):
    """Format duration in seconds to human readable format."""
    if seconds is None:
        return "Active"
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def view_all_entries(db_path: str, limit: int = 50):
    """View recent entries."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, identity, track_id, entry_time, exit_time, duration
        FROM presence_log
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"\n{'='*100}")
    print(f"RECENT ENTRIES (Last {limit})")
    print(f"{'='*100}")
    print(f"{'ID':<6} {'Identity':<20} {'Track ID':<10} {'Entry Time':<20} {'Exit Time':<20} {'Duration':<15}")
    print(f"{'-'*100}")
    
    for row in rows:
        entry_id, identity, track_id, entry_time, exit_time, duration = row
        exit_display = exit_time if exit_time else "Active"
        duration_display = format_duration(duration)
        
        print(f"{entry_id:<6} {identity:<20} {track_id:<10} {entry_time:<20} {exit_display:<20} {duration_display:<15}")
    
    print(f"{'='*100}\n")


def view_by_date(db_path: str, date: Optional[str] = None):
    """View entries for a specific date."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, identity, track_id, entry_time, exit_time, duration
        FROM presence_log
        WHERE date = ?
        ORDER BY entry_time DESC
    """, (date,))
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"\n{'='*100}")
    print(f"ENTRIES FOR {date}")
    print(f"{'='*100}")
    print(f"{'ID':<6} {'Identity':<20} {'Track ID':<10} {'Entry Time':<20} {'Exit Time':<20} {'Duration':<15}")
    print(f"{'-'*100}")
    
    if not rows:
        print("No entries found for this date.")
    else:
        for row in rows:
            entry_id, identity, track_id, entry_time, exit_time, duration = row
            # Extract only time portion
            entry_time_only = entry_time.split()[1] if ' ' in entry_time else entry_time
            exit_display = exit_time.split()[1] if exit_time and ' ' in exit_time else (exit_time if exit_time else "Active")
            duration_display = format_duration(duration)
            
            print(f"{entry_id:<6} {identity:<20} {track_id:<10} {entry_time_only:<20} {exit_display:<20} {duration_display:<15}")
    
    print(f"{'='*100}\n")


def view_by_identity(db_path: str, identity: str, date: Optional[str] = None):
    """View entries for a specific identity."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if date:
        cursor.execute("""
            SELECT id, track_id, entry_time, exit_time, duration
            FROM presence_log
            WHERE identity = ? AND date = ?
            ORDER BY entry_time DESC
        """, (identity, date))
        title = f"ENTRIES FOR {identity} ON {date}"
    else:
        cursor.execute("""
            SELECT id, track_id, entry_time, exit_time, duration, date
            FROM presence_log
            WHERE identity = ?
            ORDER BY entry_time DESC
        """, (identity,))
        title = f"ALL ENTRIES FOR {identity}"
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")
    
    if date:
        print(f"{'ID':<6} {'Track ID':<10} {'Entry Time':<20} {'Exit Time':<20} {'Duration':<15}")
    else:
        print(f"{'ID':<6} {'Track ID':<10} {'Date':<12} {'Entry Time':<20} {'Exit Time':<20} {'Duration':<15}")
    
    print(f"{'-'*100}")
    
    if not rows:
        print(f"No entries found for {identity}.")
    else:
        for row in rows:
            if date:
                entry_id, track_id, entry_time, exit_time, duration = row
                entry_time_only = entry_time.split()[1] if ' ' in entry_time else entry_time
                exit_display = exit_time.split()[1] if exit_time and ' ' in exit_time else (exit_time if exit_time else "Active")
                duration_display = format_duration(duration)
                print(f"{entry_id:<6} {track_id:<10} {entry_time_only:<20} {exit_display:<20} {duration_display:<15}")
            else:
                entry_id, track_id, entry_time, exit_time, duration, entry_date = row
                entry_time_only = entry_time.split()[1] if ' ' in entry_time else entry_time
                exit_display = exit_time.split()[1] if exit_time and ' ' in exit_time else (exit_time if exit_time else "Active")
                duration_display = format_duration(duration)
                print(f"{entry_id:<6} {track_id:<10} {entry_date:<12} {entry_time_only:<20} {exit_display:<20} {duration_display:<15}")
    
    print(f"{'='*100}\n")


def view_statistics(db_path: str, date: Optional[str] = None):
    """View statistics."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Total entries for the day
    cursor.execute("""
        SELECT COUNT(*) FROM presence_log WHERE date = ?
    """, (date,))
    total_entries = cursor.fetchone()[0]
    
    # Active entries (no exit time)
    cursor.execute("""
        SELECT COUNT(*) FROM presence_log WHERE date = ? AND exit_time IS NULL
    """, (date,))
    active_entries = cursor.fetchone()[0]
    
    # Completed entries
    completed_entries = total_entries - active_entries
    
    # Per-person statistics
    cursor.execute("""
        SELECT identity, COUNT(*) as visits, SUM(duration) as total_time
        FROM presence_log
        WHERE date = ? AND duration IS NOT NULL
        GROUP BY identity
        ORDER BY total_time DESC
    """, (date,))
    
    person_stats = cursor.fetchall()
    
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"STATISTICS FOR {date}")
    print(f"{'='*80}")
    print(f"Total Entries: {total_entries}")
    print(f"Active Entries: {active_entries}")
    print(f"Completed Entries: {completed_entries}")
    print(f"\n{'Person':<25} {'Visits':<10} {'Total Time':<20}")
    print(f"{'-'*80}")
    
    if person_stats:
        for identity, visits, total_time in person_stats:
            time_display = format_duration(total_time)
            print(f"{identity:<25} {visits:<10} {time_display:<20}")
    else:
        print("No completed entries for this date.")
    
    print(f"{'='*80}\n")


def view_active_entries(db_path: str):
    """View currently active entries."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, identity, track_id, entry_time, date
        FROM presence_log
        WHERE exit_time IS NULL
        ORDER BY entry_time DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"\n{'='*90}")
    print(f"ACTIVE ENTRIES")
    print(f"{'='*90}")
    print(f"{'ID':<6} {'Identity':<20} {'Track ID':<10} {'Date':<12} {'Entry Time':<20}")
    print(f"{'-'*90}")
    
    if not rows:
        print("No active entries.")
    else:
        for row in rows:
            entry_id, identity, track_id, entry_time, date = row
            print(f"{entry_id:<6} {identity:<20} {track_id:<10} {date:<12} {entry_time:<20}")
    
    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="View simplified presence log database")
    parser.add_argument("--db", default="presence_log.db", help="Database file path")
    parser.add_argument("--all", action="store_true", help="View all recent entries")
    parser.add_argument("--date", type=str, help="View entries for specific date (YYYY-MM-DD)")
    parser.add_argument("--identity", type=str, help="View entries for specific identity")
    parser.add_argument("--stats", action="store_true", help="View statistics")
    parser.add_argument("--active", action="store_true", help="View active entries")
    parser.add_argument("--limit", type=int, default=50, help="Limit for recent entries")
    
    args = parser.parse_args()
    
    if args.stats:
        view_statistics(args.db, args.date)
    elif args.active:
        view_active_entries(args.db)
    elif args.identity:
        view_by_identity(args.db, args.identity, args.date)
    elif args.date:
        view_by_date(args.db, args.date)
    elif args.all:
        view_all_entries(args.db, args.limit)
    else:
        # Default: show today's entries
        today = datetime.now().strftime("%Y-%m-%d")
        view_by_date(args.db, today)
        view_statistics(args.db, today)
        view_active_entries(args.db)


if __name__ == "__main__":
    main()
