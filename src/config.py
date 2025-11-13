"""
Configuration module for Sleep State Detection project.

This module contains all configuration parameters and constants used throughout
the project, making it easy to modify settings in one central location.
"""

from pathlib import Path
from typing import Dict, Any


# Model Configuration
MODEL_CONFIG: Dict[str, Any] = {
    "n_estimators": 100,
    "random_state": 42,
}

# Feature Engineering Configuration
FEATURE_CONFIG: Dict[str, Any] = {
    "window_size": 10,
    "min_periods": 1,
}

# Data Split Configuration
TRAIN_TEST_SPLIT_CONFIG: Dict[str, float] = {
    "test_size": 0.2,
    "random_state": 42,
}

# Event Label Mapping
EVENT_LABEL_MAP: Dict[str, int] = {
    "onset": 0,
    "wakeup": 1,
}

# Feature columns used for training
FEATURE_COLUMNS = [
    "hour",
    "anglez_rolling_avg",
    "enmo_rolling_avg",
]

# Default data paths (can be overridden)
DEFAULT_PATHS: Dict[str, str] = {
    "train_series": "train_series.parquet",
    "train_events": "train_events.csv",
    "test_series": "test_series.parquet",
    "submission": "submission.csv",
}


def get_data_path(key: str, base_dir: str = None) -> Path:
    """
    Get the full path for a data file.
    
    Args:
        key: Key identifying the data file (from DEFAULT_PATHS)
        base_dir: Optional base directory path. If None, returns relative path.
        
    Returns:
        Path object pointing to the data file
        
    Raises:
        KeyError: If the key is not found in DEFAULT_PATHS
    """
    if key not in DEFAULT_PATHS:
        raise KeyError(f"Unknown data path key: {key}")
    
    if base_dir:
        return Path(base_dir) / DEFAULT_PATHS[key]
    return Path(DEFAULT_PATHS[key])
