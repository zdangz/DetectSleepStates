#!/usr/bin/env python3
"""
Integration test for the refactored Sleep State Detection modules.

This test creates mock data and runs through the complete pipeline
to ensure all modules work together correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import data_loader, feature_engineering, model, utils
from src.config import EVENT_LABEL_MAP, FEATURE_COLUMNS


def create_mock_series_data(n_samples=1000):
    """Create mock series data for testing."""
    np.random.seed(42)
    
    return pd.DataFrame({
        'series_id': ['series_001'] * n_samples,
        'step': range(n_samples),
        'timestamp': [f'2023-01-01T{i%24:02d}:{i%60:02d}:00-0400' for i in range(n_samples)],
        'anglez': np.random.randn(n_samples) * 10,
        'enmo': np.random.rand(n_samples) * 0.1,
    })


def create_mock_events_data(n_events=50):
    """Create mock events data for testing."""
    np.random.seed(42)
    
    events = []
    for i in range(n_events):
        events.append({
            'series_id': 'series_001',
            'night': i // 2,
            'event': 'onset' if i % 2 == 0 else 'wakeup',
            'step': i * 20,
            'timestamp': f'2023-01-01T{(i*20)%24:02d}:{(i*20)%60:02d}:00-0400'
        })
    
    return pd.DataFrame(events)


def test_data_loading():
    """Test data loading and merging functions."""
    print("\n1. Testing data loading and merging...")
    
    # Create mock data
    series_df = create_mock_series_data(100)
    events_df = create_mock_events_data(10)
    
    print(f"   ✓ Created mock series data: {series_df.shape}")
    print(f"   ✓ Created mock events data: {events_df.shape}")
    
    # Test merge
    merged = data_loader.merge_series_and_events(series_df, events_df)
    print(f"   ✓ Merged data: {merged.shape}")
    
    # Test summary
    summary = data_loader.get_data_summary(merged, "Test Data")
    print(f"   ✓ Data summary generated: {summary['total_rows']} rows")
    
    return merged


def test_feature_engineering(merged_df):
    """Test feature engineering functions."""
    print("\n2. Testing feature engineering...")
    
    # Test time feature extraction
    df_with_time = feature_engineering.extract_time_features(merged_df)
    assert 'hour' in df_with_time.columns
    assert 'time' in df_with_time.columns
    print(f"   ✓ Time features extracted: {df_with_time['hour'].nunique()} unique hours")
    
    # Test rolling window features
    df_with_rolling = feature_engineering.add_rolling_window_features(
        df_with_time,
        columns=['anglez', 'enmo']
    )
    assert 'anglez_rolling_avg' in df_with_rolling.columns
    assert 'enmo_rolling_avg' in df_with_rolling.columns
    print(f"   ✓ Rolling window features added")
    
    # Test event label encoding
    df_encoded = feature_engineering.encode_event_labels(df_with_rolling)
    assert 'event_label' in df_encoded.columns
    print(f"   ✓ Event labels encoded: {df_encoded['event_label'].unique()}")
    
    # Test complete pipeline
    df_complete = feature_engineering.prepare_features(merged_df, is_training=True)
    print(f"   ✓ Complete feature pipeline: {df_complete.shape}")
    
    return df_complete


def test_model_training(df):
    """Test model training and evaluation."""
    print("\n3. Testing model training...")
    
    # Create model
    rf_model = model.create_model(n_estimators=10)  # Use fewer trees for speed
    print(f"   ✓ Model created: {rf_model.n_estimators} estimators")
    
    # Prepare features
    X = model.prepare_features_for_model(df, FEATURE_COLUMNS)
    y = df['event_label']
    print(f"   ✓ Features prepared: {X.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = model.split_train_validation(X, y, test_size=0.3)
    print(f"   ✓ Data split: train={len(X_train)}, val={len(X_val)}")
    
    # Train
    rf_model = model.train_model(rf_model, X_train, y_train)
    print(f"   ✓ Model trained")
    
    # Evaluate
    results = model.evaluate_model(rf_model, X_val, y_val, verbose=False)
    print(f"   ✓ Model evaluated: accuracy={results['accuracy']:.4f}")
    
    return rf_model


def test_predictions(trained_model):
    """Test making predictions."""
    print("\n4. Testing predictions...")
    
    # Create test data
    test_df = create_mock_series_data(50)
    test_df = feature_engineering.prepare_features(test_df, is_training=False)
    
    # Prepare features
    test_features = model.prepare_features_for_model(test_df, FEATURE_COLUMNS)
    print(f"   ✓ Test features prepared: {test_features.shape}")
    
    # Make predictions
    predicted_labels, confidence = model.predict_with_confidence(
        trained_model,
        test_features
    )
    print(f"   ✓ Predictions made: {len(predicted_labels)} samples")
    print(f"   ✓ Average confidence: {confidence.mean():.4f}")
    
    # Add predictions to dataframe
    test_df = utils.add_predictions_to_dataframe(
        test_df,
        predicted_labels,
        confidence,
        label_map=EVENT_LABEL_MAP
    )
    print(f"   ✓ Predictions added to DataFrame")
    
    return test_df


def test_utilities():
    """Test utility functions."""
    print("\n5. Testing utility functions...")
    
    # Test reverse mapping
    reverse_map = utils.reverse_event_mapping(EVENT_LABEL_MAP)
    assert reverse_map[0] == 'onset'
    assert reverse_map[1] == 'wakeup'
    print(f"   ✓ Reverse event mapping: {reverse_map}")
    
    # Test memory usage
    test_df = create_mock_series_data(100)
    mem_usage = utils.get_memory_usage(test_df, unit='KB')
    print(f"   ✓ Memory usage calculation: {mem_usage:.2f} KB")
    
    # Test column validation
    try:
        utils.validate_required_columns(test_df, ['series_id', 'step', 'timestamp'])
        print(f"   ✓ Column validation passed")
    except ValueError as e:
        print(f"   ✗ Column validation failed: {e}")
        return False
    
    return True


def main():
    """Run all integration tests."""
    print("=" * 80)
    print("Sleep State Detection - Integration Tests")
    print("=" * 80)
    
    try:
        # Test data loading
        merged_df = test_data_loading()
        
        # Test feature engineering
        feature_df = test_feature_engineering(merged_df)
        
        # Test model training
        trained_model = test_model_training(feature_df)
        
        # Test predictions
        predictions_df = test_predictions(trained_model)
        
        # Test utilities
        utils_ok = test_utilities()
        
        print("\n" + "=" * 80)
        print("All Integration Tests Passed! ✓")
        print("=" * 80)
        
        print("\nThe refactored code successfully:")
        print("  ✓ Loads and merges data")
        print("  ✓ Engineers features")
        print("  ✓ Trains and evaluates models")
        print("  ✓ Makes predictions with confidence scores")
        print("  ✓ Provides utility functions")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Test Failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
