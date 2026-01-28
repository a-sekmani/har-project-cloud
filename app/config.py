"""Configuration settings for the Cloud HAR application."""
import os

# API Key for authentication
API_KEY: str = os.getenv("API_KEY", "dev-key")

# Server settings
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
