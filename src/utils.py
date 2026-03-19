"""
Utility functions for ET GenAI Hackathon
"""

import logging
import json
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save dictionary to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save object to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load object from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def generate_hash(text: str) -> str:
    """Generate SHA-256 hash of text"""
    return hashlib.sha256(text.encode()).hexdigest()

def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format timestamp to ISO format"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.isoformat()

def create_directory_if_not_exists(path: Union[str, Path]) -> None:
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def validate_gstin(gstin: str) -> bool:
    """Validate GSTIN format"""
    if len(gstin) != 15:
        return False
    if not gstin[:2].isdigit():
        return False
    if not gstin[2:12].isalnum():
        return False
    if not gstin[12:13].isalpha():
        return False
    if not gstin[13:15].isdigit():
        return False
    return True

def extract_pan_from_gstin(gstin: str) -> str:
    """Extract PAN from GSTIN"""
    if len(gstin) >= 12:
        return gstin[2:12]
    return ""

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep basic punctuation
    text = ''.join(char for char in text if char.isalnum() or char.isspace() or char in '.,!?-')
    return text.strip()

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using Jaccard similarity"""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def batch_process(items: List[Any], batch_size: int = 32):
    """Yield batches of items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    if denominator == 0:
        return default
    return numerator / denominator

def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to [0, 1] range"""
    if max_val == min_val:
        return 0.0
    return (score - min_val) / (max_val - min_val)

def format_currency(amount: float, currency: str = "₹") -> str:
    """Format currency amount"""
    return f"{currency}{amount:,.2f}"

def parse_date(date_str: str, formats: List[str] = None) -> Optional[datetime]:
    """Parse date string with multiple formats"""
    if formats is None:
        formats = [
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S"
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None

class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"Starting: {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time
        logger.info(f"Completed: {self.description} in {duration.total_seconds():.2f}s")

def validate_email(email: str) -> bool:
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """Basic Indian phone number validation"""
    import re
    # Remove all non-digits
    phone = re.sub(r'\D', '', phone)
    # Check if it's 10 digits (mobile) or 11-12 digits (with country code)
    return len(phone) in [10, 11, 12] and phone.isdigit()

def get_file_size(filepath: Union[str, Path]) -> str:
    """Get human-readable file size"""
    size = Path(filepath).stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    import time
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
