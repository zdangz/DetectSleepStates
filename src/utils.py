"""
Utility functions for Sleep State Detection project.

This module contains helper functions for memory management,
data export, and other common operations.
"""

import gc
from typing import Any, List
import pandas as pd


def clear_memory(*objects: Any) -> None:
    """
    Delete objects and force garbage collection to free memory.
    
    Useful when working with large datasets to avoid memory issues.
    
    Args:
        *objects: Variable number of objects to delete
        
    Example:
        >>> import pandas as pd
        >>> df1 = pd.DataFrame({'a': [1, 2, 3]})
        >>> df2 = pd.DataFrame({'b': [4, 5, 6]})
        >>> clear_memory(df1, df2)
    """
    for obj in objects:
        del obj
    gc.collect()


def drop_columns_inplace(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drop columns from DataFrame and return it.
    
    Args:
        df: DataFrame to modify
        columns: List of column names to drop
        
    Returns:
        DataFrame with columns dropped
        
    Note:
        Only drops columns that exist in the DataFrame.
        Missing columns are silently ignored.
    """
    existing_cols = [col for col in columns if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
    return df


def reverse_event_mapping(label_map: dict) -> dict:
    """
    Create reverse mapping from event labels to event names.
    
    Args:
        label_map: Dictionary mapping event names to integer labels
        
    Returns:
        Dictionary mapping integer labels to event names
        
    Example:
        >>> from src.config import EVENT_LABEL_MAP
        >>> reverse_map = reverse_event_mapping(EVENT_LABEL_MAP)
        >>> print(reverse_map)
        {0: 'onset', 1: 'wakeup'}
    """
    return {v: k for k, v in label_map.items()}


def add_predictions_to_dataframe(
    df: pd.DataFrame,
    predicted_labels: Any,
    confidence_scores: Any,
    label_column: str = 'predicted_event_label',
    confidence_column: str = 'confidence_score',
    event_column: str = 'event',
    label_map: dict = None
) -> pd.DataFrame:
    """
    Add prediction results to a DataFrame.
    
    Args:
        df: DataFrame to add predictions to
        predicted_labels: Array of predicted integer labels
        confidence_scores: Array of confidence scores
        label_column: Name for predicted label column
        confidence_column: Name for confidence score column
        event_column: Name for event name column (maps labels to names)
        label_map: Dictionary to map labels to event names. If None, skips event mapping.
        
    Returns:
        DataFrame with prediction columns added
    """
    df = df.copy()
    df[label_column] = predicted_labels
    df[confidence_column] = confidence_scores
    
    if label_map is not None:
        reverse_map = reverse_event_mapping(label_map)
        df[event_column] = df[label_column].map(reverse_map)
    
    return df


def export_submission(
    df: pd.DataFrame,
    output_path: str,
    columns: List[str] = None
) -> None:
    """
    Export predictions to CSV file for submission.
    
    Args:
        df: DataFrame containing predictions
        output_path: Path where CSV file should be saved
        columns: List of columns to include. If None, includes all columns.
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        if columns:
            df = df[columns]
        df.to_csv(output_path, index=False)
        print(f"Submission file saved to: {output_path}")
    except Exception as e:
        raise IOError(f"Error writing submission file: {str(e)}")


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    df_name: str = "DataFrame"
) -> None:
    """
    Validate that a DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name of DataFrame for error messages
        
    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {', '.join(missing)}"
        )


def get_memory_usage(df: pd.DataFrame, unit: str = 'MB') -> float:
    """
    Get memory usage of a DataFrame.
    
    Args:
        df: DataFrame to measure
        unit: Unit for memory size ('B', 'KB', 'MB', 'GB')
        
    Returns:
        Memory usage in specified unit
        
    Raises:
        ValueError: If unit is not recognized
    """
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    
    if unit not in units:
        raise ValueError(f"Unit must be one of {list(units.keys())}")
    
    bytes_used = df.memory_usage(deep=True).sum()
    return bytes_used / units[unit]
