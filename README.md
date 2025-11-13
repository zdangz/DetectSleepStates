# DetectSleepStates

A machine learning project for detecting sleep states using accelerometer data.

## Project Structure

The project has been refactored into a modular structure for better code organization and reusability:

```
DetectSleepStates/
├── src/                          # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration and constants
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── feature_engineering.py   # Feature extraction and engineering
│   ├── model.py                  # Model training and prediction
│   └── utils.py                  # Utility functions
├── ML - Sleep State Detection - Refactored.ipynb  # Refactored notebook
├── ML - Sleep State Detection.ipynb               # Original notebook
├── randomforestclassifier.ipynb                   # Original RF notebook
└── README.md                     # This file
```

## Modules Overview

### `src/config.py`
Contains all configuration parameters and constants:
- Model hyperparameters (n_estimators, random_state)
- Feature engineering parameters (window_size)
- Event label mappings
- Default data paths

### `src/data_loader.py`
Handles data loading and preprocessing:
- `load_train_data()`: Load training series and events
- `load_test_data()`: Load test series
- `merge_series_and_events()`: Merge datasets
- `get_data_summary()`: Get dataset statistics

### `src/feature_engineering.py`
Feature extraction and engineering:
- `extract_time_features()`: Extract hour, time from timestamps
- `add_rolling_window_features()`: Calculate rolling averages
- `encode_event_labels()`: Convert event strings to numeric labels
- `prepare_features()`: Complete feature engineering pipeline

### `src/model.py`
Model training and prediction:
- `create_model()`: Initialize Random Forest classifier
- `train_and_evaluate_pipeline()`: Complete training pipeline
- `predict_with_confidence()`: Make predictions with confidence scores
- `evaluate_model()`: Evaluate model performance

### `src/utils.py`
Utility functions:
- `clear_memory()`: Memory management
- `add_predictions_to_dataframe()`: Add predictions to results
- `export_submission()`: Export predictions to CSV
- `get_memory_usage()`: Monitor DataFrame memory usage

## Usage

### Using the Refactored Modules

```python
import sys
sys.path.insert(0, '.')

from src import data_loader, feature_engineering, model
from src.config import EVENT_LABEL_MAP

# Load data
train_series, train_events = data_loader.load_train_data(
    "path/to/train_series.parquet",
    "path/to/train_events.csv"
)

# Merge and engineer features
merged_data = data_loader.merge_series_and_events(train_series, train_events)
merged_data = feature_engineering.prepare_features(merged_data, is_training=True)

# Train and evaluate
trained_model, results = model.train_and_evaluate_pipeline(merged_data)

# Make predictions on test data
test_series = data_loader.load_test_data("path/to/test_series.parquet")
test_series = feature_engineering.prepare_features(test_series, is_training=False)
predictions, confidence = model.predict_with_confidence(trained_model, test_series)
```

### Using the Refactored Notebook

Open `ML - Sleep State Detection - Refactored.ipynb` for a complete example that uses all the refactored modules.

## Benefits of the Refactored Structure

1. **Modularity**: Code is organized into logical, reusable modules
2. **Maintainability**: Easier to update and fix issues in specific components
3. **Readability**: Clear separation of concerns with descriptive function names
4. **Documentation**: Comprehensive docstrings for all functions
5. **Configurability**: Centralized configuration for easy parameter tuning
6. **Error Handling**: Proper error handling and validation
7. **Type Safety**: Type hints for better code clarity
8. **Testability**: Modular functions are easier to unit test

## References

Exploratory Data Analysis:
- https://www.kaggle.com/code/dumisanisibanda/exploratory-accelerometer-data-analysis

EDA Plots:
- https://www.kaggle.com/code/yihsuankao/sleep-eda-plots

Random Forest Starter:
- https://www.kaggle.com/code/sumitai/zzzs-random-forest-model-starter

## Changelog

### Current Version
- Refactored code into modular structure with src/ package
- Added comprehensive documentation and type hints
- Centralized configuration management
- Improved error handling and validation
- Created refactored example notebook

### 28 October 2023
- Fixed "Notebook out of memory" issue during submission by deleting unused dataframes and importing gc
- To debug "Submission scoring error"
