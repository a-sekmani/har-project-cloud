#!/usr/bin/env python3
"""
Empty all application tables (devices, activity_events, pose_windows).
Schema and migrations are left intact.

Usage:
  python scripts/clear_db.py
  DATABASE_URL=postgresql+psycopg://... python scripts/clear_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from app.database import engine


def main():
    # Truncate all in one statement so PostgreSQL accepts FK references between tables
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE activity_events, pose_windows, devices"))
        conn.commit()
    print("Truncated: activity_events, pose_windows, devices")
    print("Database cleared.")


if __name__ == "__main__":
    main()
