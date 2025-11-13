"""
Data loading and preprocessing module for Sleep State Detection.

This module handles loading data from various sources and performing
initial data cleaning operations.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd


def load_train_data(
    series_path: str,
    events_path: str,
    clean_na: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training series and events data.
    
    Args:
        series_path: Path to the training series parquet file
        events_path: Path to the training events CSV file
        clean_na: If True, remove rows with NaN values from events data
        
    Returns:
        Tuple of (train_series DataFrame, train_events DataFrame)
        
    Raises:
        FileNotFoundError: If either file doesn't exist
        pd.errors.ParserError: If file parsing fails
    """
    try:
        train_series = pd.read_parquet(series_path)
        train_events = pd.read_csv(events_path)
        
        if clean_na:
            train_events = train_events.dropna()
            
        return train_series, train_events
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e.filename}")
    except Exception as e:
        raise RuntimeError(f"Error loading training data: {str(e)}")


def load_test_data(series_path: str) -> pd.DataFrame:
    """
    Load test series data.
    
    Args:
        series_path: Path to the test series parquet file
        
    Returns:
        Test series DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.ParserError: If file parsing fails
    """
    try:
        return pd.read_parquet(series_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test data file not found: {series_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading test data: {str(e)}")


def merge_series_and_events(
    series_df: pd.DataFrame,
    events_df: pd.DataFrame,
    on_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Merge series and events dataframes.
    
    Args:
        series_df: Series DataFrame containing sensor data
        events_df: Events DataFrame containing sleep event labels
        on_columns: List of column names to merge on. 
                   Defaults to ['series_id', 'step', 'timestamp']
        
    Returns:
        Merged DataFrame
        
    Raises:
        ValueError: If required merge columns are missing
    """
    if on_columns is None:
        on_columns = ['series_id', 'step', 'timestamp']
    
    # Validate that merge columns exist in both dataframes
    for col in on_columns:
        if col not in series_df.columns:
            raise ValueError(f"Column '{col}' not found in series DataFrame")
        if col not in events_df.columns:
            raise ValueError(f"Column '{col}' not found in events DataFrame")
    
    return pd.merge(series_df, events_df, on=on_columns)


def get_data_summary(df: pd.DataFrame, name: str = "DataFrame") -> dict:
    """
    Get summary statistics for a DataFrame.
    
    Args:
        df: DataFrame to summarize
        name: Name of the DataFrame for display purposes
        
    Returns:
        Dictionary containing summary statistics
    """
    nan_count = df.isna().any(axis=1).sum()
    
    return {
        "name": name,
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "nan_rows": nan_count,
        "columns": list(df.columns),
    }


def print_data_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print summary statistics for a DataFrame.
    
    Args:
        df: DataFrame to summarize
        name: Name of the DataFrame for display purposes
    """
    summary = get_data_summary(df, name)
    print(f"\n{summary['name']} Summary:")
    print(f"  Total rows: {summary['total_rows']}")
    print(f"  Total columns: {summary['total_columns']}")
    print(f"  Rows with NaN: {summary['nan_rows']}")
    print(f"  Columns: {', '.join(summary['columns'])}")
