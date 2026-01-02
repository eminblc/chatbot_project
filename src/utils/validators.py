"""Input validation utilities."""
import re
from pathlib import Path


def validate_file_path(file_path, must_exist=True, allowed_extensions=None):
    """Validate file path and extensions."""
    if not file_path:
        raise ValueError("File path cannot be empty")
    path = Path(file_path)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if allowed_extensions and path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        raise ValueError(f"Invalid extension. Allowed: {', '.join(allowed_extensions)}")
    return path


def validate_query(query, min_length=3, max_length=1000):
    """Validate query string length."""
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    query = query.strip()
    if len(query) < min_length:
        raise ValueError(f"Query must be at least {min_length} characters long")
    if len(query) > max_length:
        raise ValueError(f"Query must not exceed {max_length} characters")
    return True


def sanitize_filename(filename):
    """Sanitize filename: remove invalid chars."""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip('. ')
    return filename if filename else "untitled"
