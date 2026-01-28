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
# Default: Local PostgreSQL instance
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://cloudhar:cloudhar@localhost:5432/cloudhar"
)
