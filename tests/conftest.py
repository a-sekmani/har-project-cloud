"""
Pytest configuration. Set DATABASE_URL to SQLite before app is imported
so that app.database uses SQLite and handle_completed_window (Phase 3) writes
to the same DB that tests can query.
"""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
