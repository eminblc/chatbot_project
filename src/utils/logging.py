"""Logging utilities."""
from typing import Optional
import logging
import sys
from pathlib import Path

def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with console and optional file handlers."""
    logger = logging.getLogger("chatbot")
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(f"chatbot.{name}" if name else "chatbot")
