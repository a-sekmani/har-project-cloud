"""
Configuration settings for the Cloud HAR application.

This module centralizes all configuration settings, reading from environment
variables with sensible defaults. This makes the application easy to configure
for different environments (development, staging, production).

All settings can be overridden via environment variables.
"""
import os

# API Key for authentication
# Used to secure the inference endpoint
# Default: "dev-key" (for development only)
# Production: Should be set via environment variable
API_KEY: str = os.getenv("API_KEY", "dev-key")

# Server settings
# HOST: IP address to bind to (0.0.0.0 = all interfaces)
# PORT: Port number to listen on
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# Logging
# LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Database
# DATABASE_URL: PostgreSQL connection string
# Format: postgresql+psycopg://user:password@host:port/database
# Default: localhost:5433 (matches docker-compose postgres port mapping 5433:5432)
# If you run Postgres directly on host, set DATABASE_URL with port 5432 (or your port).
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://cloudhar:cloudhar@localhost:5433/cloudhar"
)

# Edge aggregation (frame_event -> window payload bridge)
# EDGE_WINDOW_SIZE: number of frames per window (default 30, ≈1 s at 30 fps)
EDGE_WINDOW_SIZE: int = int(os.getenv("EDGE_WINDOW_SIZE", "30"))
# EDGE_CAMERA_ID_DEFAULT: used when source.camera_id / header / query are missing
EDGE_CAMERA_ID_DEFAULT: str = os.getenv("EDGE_CAMERA_ID_DEFAULT", "cam-1")
# EDGE_AUTO_INFER: when True, run infer_and_persist on completed window (edge -> DB)
EDGE_AUTO_INFER: bool = os.getenv("EDGE_AUTO_INFER", "0").lower() in ("1", "true", "yes")
