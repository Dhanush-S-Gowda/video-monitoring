"""
Database Viewer Utility
View and query the presence log database.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional
import argparse
from pathlib import Path


class DatabaseViewer:
    """Utility to view and query presence log database."""
    
    def __init__(self, db_path: str = "presence_log.db"):
        """Initialize database viewer."""
        self.db_path = db_path
        
        if not Path(db_path).exists():
            print(f"Error: Database file '{db_path}' not found!")
            print("Run the integrated system first to create the database.")
            exit(1)
    
    def execute_query(self, query: str, params: tuple = ()):
        """Execute a query and return results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        return results
    
    def print_table(self, headers: list, rows: list):
        """Print a formatted table."""
        if not rows:
            print("No data found.")
            return
        
        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Print header
        header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
        print("\n" + header_line)
        print("-" * len(header_line))
        
        # Print rows
        for row in rows:
            print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
        
        print()
    
    def show_all_sessions(self, limit: int = 50):
        """Show all sessions in the database."""
        print(f"\n{'=' * 80}")
        print(f"ALL SESSIONS (Last {limit})")
        print("=" * 80)
        
        query = """
            SELECT id, date, user, start_time, end_time, 
                   COALESCE(duration, 0) as duration, status
            FROM presence_log
            ORDER BY date DESC, start_time DESC
            LIMIT ?
        """
        results = self.execute_query(query, (limit,))
        
        headers = ["ID", "Date", "User", "Start", "End", "Duration(s)", "Status"]
        self.print_table(headers, results)
    
    def show_today_sessions(self):
        """Show today's sessions."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        print(f"\n{'=' * 80}")
        print(f"TODAY'S SESSIONS ({today})")
        print("=" * 80)
        
        query = """
            SELECT id, user, start_time, end_time, 
                   COALESCE(duration, 0) as duration, status
            FROM presence_log
            WHERE date = ?
            ORDER BY start_time DESC
        """
        results = self.execute_query(query, (today,))
        
        headers = ["ID", "User", "Start", "End", "Duration(s)", "Status"]
        self.print_table(headers, results)
    
    def show_active_sessions(self):
        """Show currently active sessions."""
        print(f"\n{'=' * 80}")
        print("ACTIVE SESSIONS")
        print("=" * 80)
        
        query = """
            SELECT id, date, user, start_time, end_time, track_id
            FROM presence_log
            WHERE status = 'active'
            ORDER BY id DESC
        """
        results = self.execute_query(query)
        
        headers = ["ID", "Date", "User", "Start", "Last Seen", "Track ID"]
        self.print_table(headers, results)
    
    def show_user_stats(self, date: Optional[str] = None):
        """Show statistics per user."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"\n{'=' * 80}")
        print(f"USER STATISTICS ({date})")
        print("=" * 80)
        
        query = """
            SELECT 
                user,
                COUNT(*) as sessions,
                SUM(COALESCE(duration, 0)) as total_seconds,
                AVG(COALESCE(duration, 0)) as avg_seconds,
                MIN(start_time) as first_seen,
                MAX(end_time) as last_seen
            FROM presence_log
            WHERE date = ?
            GROUP BY user
            ORDER BY total_seconds DESC
        """
        results = self.execute_query(query, (date,))
        
        # Format results to show time in HH:MM:SS
        formatted_results = []
        for row in results:
            user, sessions, total_sec, avg_sec, first, last = row
            total_formatted = self._format_seconds(total_sec)
            avg_formatted = self._format_seconds(avg_sec)
            formatted_results.append([user, sessions, total_formatted, avg_formatted, first, last])
        
        headers = ["User", "Sessions", "Total Time", "Avg Time", "First Seen", "Last Seen"]
        self.print_table(headers, formatted_results)
    
    def show_user_sessions(self, username: str, days: int = 7):
        """Show sessions for a specific user."""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        print(f"\n{'=' * 80}")
        print(f"SESSIONS FOR USER: {username} (Last {days} days)")
        print("=" * 80)
        
        query = """
            SELECT id, date, start_time, end_time, 
                   COALESCE(duration, 0) as duration, status
            FROM presence_log
            WHERE user = ? AND date >= ?
            ORDER BY date DESC, start_time DESC
        """
        results = self.execute_query(query, (username, cutoff_date))
        
        # Format duration
        formatted_results = []
        for row in results:
            id_, date, start, end, duration, status = row
            duration_formatted = self._format_seconds(duration)
            formatted_results.append([id_, date, start, end, duration_formatted, status])
        
        headers = ["ID", "Date", "Start", "End", "Duration", "Status"]
        self.print_table(headers, formatted_results)
    
    def show_summary(self):
        """Show database summary."""
        print(f"\n{'=' * 80}")
        print("DATABASE SUMMARY")
        print("=" * 80)
        
        # Total sessions
        total = self.execute_query("SELECT COUNT(*) FROM presence_log")[0][0]
        print(f"\nTotal Sessions: {total}")
        
        # Active sessions
        active = self.execute_query("SELECT COUNT(*) FROM presence_log WHERE status = 'active'")[0][0]
        print(f"Active Sessions: {active}")
        
        # Unique users
        users = self.execute_query("SELECT COUNT(DISTINCT user) FROM presence_log")[0][0]
        print(f"Unique Users: {users}")
        
        # Date range
        date_range = self.execute_query("""
            SELECT MIN(date), MAX(date) FROM presence_log
        """)
        if date_range[0][0]:
            print(f"Date Range: {date_range[0][0]} to {date_range[0][1]}")
        
        # Most active user
        most_active = self.execute_query("""
            SELECT user, COUNT(*) as sessions
            FROM presence_log
            GROUP BY user
            ORDER BY sessions DESC
            LIMIT 1
        """)
        if most_active:
            print(f"Most Active User: {most_active[0][0]} ({most_active[0][1]} sessions)")
        
        print()
    
    def _format_seconds(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        if seconds is None:
            return "00:00:00"
        
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def custom_query(self, query: str):
        """Execute a custom SQL query."""
        print(f"\n{'=' * 80}")
        print("CUSTOM QUERY RESULTS")
        print("=" * 80)
        print(f"Query: {query}\n")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Get column names
            headers = [description[0] for description in cursor.description]
            
            conn.close()
            
            self.print_table(headers, results)
        except Exception as e:
            print(f"Error executing query: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="View and query presence log database")
    parser.add_argument("--db", type=str, default="presence_log.db",
                       help="Path to database file")
    parser.add_argument("--all", action="store_true",
                       help="Show all sessions")
    parser.add_argument("--today", action="store_true",
                       help="Show today's sessions")
    parser.add_argument("--active", action="store_true",
                       help="Show active sessions")
    parser.add_argument("--stats", action="store_true",
                       help="Show user statistics for today")
    parser.add_argument("--user", type=str,
                       help="Show sessions for specific user")
    parser.add_argument("--days", type=int, default=7,
                       help="Number of days to show (for --user)")
    parser.add_argument("--summary", action="store_true",
                       help="Show database summary")
    parser.add_argument("--query", type=str,
                       help="Execute custom SQL query")
    parser.add_argument("--limit", type=int, default=50,
                       help="Limit number of results")
    
    args = parser.parse_args()
    
    viewer = DatabaseViewer(db_path=args.db)
    
    # If no specific command, show summary and today's sessions
    if not any([args.all, args.today, args.active, args.stats, args.user, args.summary, args.query]):
        viewer.show_summary()
        viewer.show_today_sessions()
        return
    
    # Execute requested commands
    if args.summary:
        viewer.show_summary()
    
    if args.all:
        viewer.show_all_sessions(limit=args.limit)
    
    if args.today:
        viewer.show_today_sessions()
    
    if args.active:
        viewer.show_active_sessions()
    
    if args.stats:
        viewer.show_user_stats()
    
    if args.user:
        viewer.show_user_sessions(args.user, days=args.days)
    
    if args.query:
        viewer.custom_query(args.query)


if __name__ == "__main__":
    main()
