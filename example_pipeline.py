#!/usr/bin/env python3
"""
Example script demonstrating the refactored Sleep State Detection pipeline.

This script shows how to use the modular components to build a complete
sleep state detection workflow.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import data_loader, feature_engineering, model, utils
from src.config import EVENT_LABEL_MAP


def main():
    """Run the complete sleep state detection pipeline."""
    
    print("=" * 80)
    print("Sleep State Detection Pipeline - Refactored")
    print("=" * 80)
    
    # Configure paths (update these to your actual data paths)
    TRAIN_SERIES_PATH = "path/to/train_series.parquet"
    TRAIN_EVENTS_PATH = "path/to/train_events.csv"
    TEST_SERIES_PATH = "path/to/test_series.parquet"
    SUBMISSION_PATH = "submission_output.csv"
    
    print("\nNote: Update data paths in this script to point to your actual data files.")
    print("\nThis is a demonstration of the refactored code structure.")
    print("The following steps would be executed with actual data:\n")
    
    # Step 1: Load training data
    print("1. Load training data")
    print("   - data_loader.load_train_data()")
    print("   - Automatically cleans NaN values")
    print("   - Returns train_series and train_events DataFrames")
    
    # Step 2: Merge datasets
    print("\n2. Merge series and events")
    print("   - data_loader.merge_series_and_events()")
    print("   - Merges on series_id, step, and timestamp")
    print("   - Returns merged DataFrame")
    
    # Step 3: Feature engineering
    print("\n3. Feature engineering")
    print("   - feature_engineering.prepare_features()")
    print("   - Extracts time features (hour, time)")
    print("   - Adds rolling window averages for anglez and enmo")
    print("   - Encodes event labels to numeric values")
    
    # Step 4: Train model
    print("\n4. Train and evaluate model")
    print("   - model.train_and_evaluate_pipeline()")
    print("   - Creates Random Forest classifier")
    print("   - Splits data into train/validation")
    print("   - Trains model and reports accuracy")
    
    # Step 5: Test predictions
    print("\n5. Process test data")
    print("   - data_loader.load_test_data()")
    print("   - feature_engineering.prepare_features(is_training=False)")
    print("   - model.predict_with_confidence()")
    
    # Step 6: Export results
    print("\n6. Export predictions")
    print("   - utils.add_predictions_to_dataframe()")
    print("   - utils.export_submission()")
    
    print("\n" + "=" * 80)
    print("Pipeline Overview Complete")
    print("=" * 80)
    print("\nKey Benefits of Refactored Code:")
    print("  ✓ Modular: Each component has a single responsibility")
    print("  ✓ Reusable: Functions can be used in different contexts")
    print("  ✓ Documented: Comprehensive docstrings for all functions")
    print("  ✓ Configurable: Central configuration in config.py")
    print("  ✓ Maintainable: Easy to update and test individual components")
    print("  ✓ Readable: Clear function names and type hints")
    
    print("\nFor a working example with actual data, see:")
    print("  - ML - Sleep State Detection - Refactored.ipynb")
    print("\nFor module documentation, check:")
    print("  - src/config.py")
    print("  - src/data_loader.py")
    print("  - src/feature_engineering.py")
    print("  - src/model.py")
    print("  - src/utils.py")
    print()


if __name__ == "__main__":
    main()
