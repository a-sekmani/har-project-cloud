"""Logging configuration for the Cloud HAR application."""
import logging
import sys
from typing import Optional
import uuid
import time
from contextlib import contextmanager

from app.config import LOG_LEVEL

# Configure root logger
logger = logging.getLogger("cloud_har")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# Create console handler if not exists
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(f"cloud_har.{name}")


@contextmanager
def log_inference_request(
    device_id: str,
    camera_id: str,
    num_people: int,
    request_id: Optional[str] = None
):
    """
    Context manager to log inference request with latency tracking.
    
    Usage:
        with log_inference_request(device_id, camera_id, num_people, request_id):
            # inference logic here
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    start_time = time.time()
    
    try:
        yield request_id
    finally:
        latency_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"inference_request | "
            f"request_id={request_id} | "
            f"device_id={device_id} | "
            f"camera_id={camera_id} | "
            f"num_people={num_people} | "
            f"latency_ms={latency_ms}"
        )
