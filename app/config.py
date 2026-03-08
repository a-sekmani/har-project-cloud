"""Configuration settings for the Cloud HAR application."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (parent of app/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# API Key for authentication
API_KEY: str = os.getenv("API_KEY", "dev-key")

# Server settings
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Database
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://cloudhar:cloudhar@localhost:5433/cloudhar",
)

# Models (for dashboard/label)
MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
MODEL_KEY_DEFAULT: str = os.getenv("MODEL_KEY_DEFAULT", "edge17_v6_lowlr")
