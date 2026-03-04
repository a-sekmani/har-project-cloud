"""
Database connection and session management.

This module handles all database-related setup including:
- SQLAlchemy engine creation
- Session factory configuration
- Base class for ORM models
- Database session dependency for FastAPI
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from app.config import DATABASE_URL


@compiles(PG_UUID, "sqlite")
def compile_uuid_sqlite(type_, compiler, **kw):
    """Render PostgreSQL UUID as TEXT for SQLite (no native UUID type)."""
    return "TEXT"

# SQLite needs check_same_thread=False for FastAPI
_connect_args = {}
if "sqlite" in DATABASE_URL.lower():
    _connect_args["check_same_thread"] = False

# Create SQLAlchemy engine
# pool_pre_ping=True: Verifies connections before using them (handles stale connections)
# echo=False: Disable SQL query logging (set to True for debugging)
engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=("sqlite" not in DATABASE_URL.lower()),
    echo=False
)

# Create session factory
# autocommit=False: Manual transaction control
# autoflush=False: Manual flush control (better performance)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all ORM models
# All models should inherit from this Base class
Base = declarative_base()


def get_db():
    """
    FastAPI dependency to get a database session.
    
    This function is used as a dependency in FastAPI endpoints to provide
    database sessions. It ensures proper session lifecycle management:
    - Creates a new session for each request
    - Yields the session to the endpoint
    - Closes the session after the request completes (even if an error occurs)
    
    Usage:
        @app.get("/endpoint")
        async def my_endpoint(db: Session = Depends(get_db)):
            # Use db session here
            pass
    
    Yields:
        Session: SQLAlchemy database session
    
    Note:
        The session is automatically closed in the finally block, ensuring
        no connection leaks even if exceptions occur.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
