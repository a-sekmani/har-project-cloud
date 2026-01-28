"""
Logging configuration for the Cloud HAR application.

This module provides structured logging functionality including:
- Logger configuration with configurable log levels
- Request logging with latency tracking
- Structured log format for easy parsing

All logs are written to stdout in a structured format that includes
timestamps, log levels, and contextual information.
"""
import logging
import sys
from typing import Optional
import uuid
import time
from contextlib import contextmanager

from app.config import LOG_LEVEL

# Configure root logger for the application
# All loggers will be children of this logger
logger = logging.getLogger("cloud_har")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# Create console handler if not exists (prevents duplicate handlers)
# This handler writes logs to stdout
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    
    # Create formatter with timestamp, logger name, level, and message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Creates a child logger under the "cloud_har" root logger.
    This allows for hierarchical logging and easy filtering.
    
    Args:
        name: Name of the logger (e.g., "main", "services")
    
    Returns:
        logging.Logger: Logger instance with name "cloud_har.{name}"
    
    Example:
        logger = get_logger("main")
        logger.info("Application started")
    """
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
    
    This context manager automatically logs inference requests with structured
    information including request ID, device ID, camera ID, number of people,
    and request latency. The latency is calculated automatically from the
    time the context is entered until it exits.
    
    Usage:
        with log_inference_request(device_id, camera_id, num_people, request_id):
            # inference logic here
            # Latency is automatically calculated and logged
    
    Args:
        device_id: Unique identifier of the device
        camera_id: Identifier of the camera
        num_people: Number of people detected in the request
        request_id: Optional request ID (generated if not provided)
    
    Yields:
        str: The request ID (for use within the context)
    
    Example:
        with log_inference_request("pi-001", "cam-1", 2, "req-123"):
            # Process inference
            pass
        # Logs: inference_request | request_id=req-123 | device_id=pi-001 | ...
    """
    # Generate request ID if not provided
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    # Record start time for latency calculation
    start_time = time.time()
    
    try:
        # Yield request ID to the context
        yield request_id
    finally:
        # Calculate latency in milliseconds
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log structured information
        logger.info(
            f"inference_request | "
            f"request_id={request_id} | "
            f"device_id={device_id} | "
            f"camera_id={camera_id} | "
            f"num_people={num_people} | "
            f"latency_ms={latency_ms}"
        )
