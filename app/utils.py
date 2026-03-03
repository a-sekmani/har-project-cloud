"""Shared utilities."""
from datetime import datetime
from typing import Optional


def isoformat_utc(dt: Optional[datetime]) -> Optional[str]:
    """
    Serialize datetime to ISO string with explicit UTC for correct browser display.

    When the value is naive (no tzinfo), appends 'Z' so that JavaScript's Date
    interprets it as UTC and converts to the user's local time correctly.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.isoformat() + "Z"
    return dt.isoformat()
