"""
Feature engineering module for Sleep State Detection.

This module contains functions for extracting and creating features
from the raw sensor data, including time-based features and rolling
window statistics.
"""

from typing import Optional
import pandas as pd

from .config import FEATURE_CONFIG, EVENT_LABEL_MAP


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from timestamp column.
    
    Extracts hour, time, and converts timestamp to UTC timezone-aware datetime.
    Removes timezone suffix from timestamp string before conversion.
    
    Args:
        df: DataFrame containing 'timestamp' column
        
    Returns:
        DataFrame with added time features:
        - timestamp_notimezone: Timestamp without timezone suffix
        - timestamp_utc: Parsed datetime in UTC
        - time: Time of day
        - hour: Hour of day (0-23)
        
    Raises:
        KeyError: If 'timestamp' column is missing
        ValueError: If timestamp format is invalid
    """
    if 'timestamp' not in df.columns:
        raise KeyError("DataFrame must contain 'timestamp' column")
    
    df = df.copy()
    
    try:
        # Remove timezone information from the timestamp (e.g., "-0400")
        df['timestamp_notimezone'] = df['timestamp'].str[:-5]
        
        # Convert timestamp to datetime in UTC
        df['timestamp_utc'] = pd.to_datetime(
            df['timestamp_notimezone'],
            utc=True
        )
        
        # Extract time from timestamp
        df['time'] = df['timestamp_utc'].dt.time
        
        # Extract hour from timestamp
        df['hour'] = df['timestamp_utc'].dt.hour
        
        return df
    except Exception as e:
        raise ValueError(f"Error extracting time features: {str(e)}")


def add_rolling_window_features(
    df: pd.DataFrame,
    columns: list,
    window_size: Optional[int] = None,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Add rolling window average features for specified columns.
    
    Args:
        df: DataFrame containing columns to process
        columns: List of column names to calculate rolling averages for
        window_size: Size of the rolling window. If None, uses config value.
        min_periods: Minimum number of observations required. If None, uses config value.
        
    Returns:
        DataFrame with added rolling average columns.
        New columns will be named '{original_column}_rolling_avg'
        
    Raises:
        KeyError: If any specified column is missing
    """
    if window_size is None:
        window_size = FEATURE_CONFIG["window_size"]
    if min_periods is None:
        min_periods = FEATURE_CONFIG["min_periods"]
    
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        
        rolling_col_name = f"{col}_rolling_avg"
        df[rolling_col_name] = df[col].rolling(
            window=window_size,
            min_periods=min_periods
        ).mean()
    
    return df


def encode_event_labels(
    df: pd.DataFrame,
    event_column: str = 'event',
    label_column: str = 'event_label',
    label_map: Optional[dict] = None
) -> pd.DataFrame:
    """
    Convert event strings to numerical labels.
    
    Args:
        df: DataFrame containing event column
        event_column: Name of the column containing event strings
        label_column: Name for the new label column
        label_map: Dictionary mapping event strings to integers.
                  If None, uses default EVENT_LABEL_MAP from config.
        
    Returns:
        DataFrame with added event_label column
        
    Raises:
        KeyError: If event_column is missing
        ValueError: If event contains values not in label_map
    """
    if event_column not in df.columns:
        raise KeyError(f"Column '{event_column}' not found in DataFrame")
    
    if label_map is None:
        label_map = EVENT_LABEL_MAP
    
    df = df.copy()
    
    # Check for unmapped values
    unique_events = df[event_column].unique()
    unmapped = [e for e in unique_events if pd.notna(e) and e not in label_map]
    if unmapped:
        raise ValueError(f"Unmapped event values found: {unmapped}")
    
    df[label_column] = df[event_column].map(label_map)
    
    return df


def prepare_features(
    df: pd.DataFrame,
    is_training: bool = True,
    sensor_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a DataFrame.
    
    This is a convenience function that applies the complete feature
    engineering pipeline in the correct order.
    
    Args:
        df: Raw DataFrame to process
        is_training: If True, also encode event labels
        sensor_columns: List of sensor columns for rolling features.
                       Defaults to ['anglez', 'enmo']
        
    Returns:
        DataFrame with all engineered features
    """
    if sensor_columns is None:
        sensor_columns = ['anglez', 'enmo']
    
    # Extract time features
    df = extract_time_features(df)
    
    # Sort by time for proper rolling window calculation
    df = df.sort_values(by='time')
    
    # Add rolling window features
    df = add_rolling_window_features(df, sensor_columns)
    
    # Encode event labels if this is training data
    if is_training and 'event' in df.columns:
        df = encode_event_labels(df)
    
    return df
