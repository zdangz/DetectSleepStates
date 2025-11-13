# Quick Start Guide

This guide will help you get started with the refactored Sleep State Detection code.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zdangz/DetectSleepStates.git
cd DetectSleepStates
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
DetectSleepStates/
├── src/                          # Main source code package
│   ├── config.py                 # Configuration and constants
│   ├── data_loader.py            # Data loading functions
│   ├── feature_engineering.py   # Feature extraction
│   ├── model.py                  # Model training/prediction
│   └── utils.py                  # Utility functions
├── ML - Sleep State Detection - Refactored.ipynb  # Example notebook
├── example_pipeline.py           # Example Python script
├── test_integration.py           # Integration tests
└── requirements.txt              # Python dependencies
```

## Quick Examples

### Example 1: Using the Refactored Notebook

Open and run `ML - Sleep State Detection - Refactored.ipynb` in Jupyter:

```bash
jupyter notebook "ML - Sleep State Detection - Refactored.ipynb"
```

Remember to update the data paths in the notebook to point to your data files.

### Example 2: Using the Modules Directly

```python
import sys
sys.path.insert(0, '.')

from src import data_loader, feature_engineering, model

# Load data
train_series, train_events = data_loader.load_train_data(
    "path/to/train_series.parquet",
    "path/to/train_events.csv"
)

# Merge and prepare features
merged = data_loader.merge_series_and_events(train_series, train_events)
merged = feature_engineering.prepare_features(merged, is_training=True)

# Train model
trained_model, results = model.train_and_evaluate_pipeline(merged)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Example 3: Running the Example Pipeline

```bash
python3 example_pipeline.py
```

This displays the pipeline structure and available functions.

### Example 4: Running Integration Tests

```bash
python3 test_integration.py
```

This runs comprehensive tests with mock data to verify all modules work correctly.

## Configuration

All configuration parameters are centralized in `src/config.py`:

- **Model parameters**: `MODEL_CONFIG` (n_estimators, random_state)
- **Feature parameters**: `FEATURE_CONFIG` (window_size, min_periods)
- **Event mappings**: `EVENT_LABEL_MAP` (onset=0, wakeup=1)
- **Feature columns**: `FEATURE_COLUMNS` (hour, anglez_rolling_avg, enmo_rolling_avg)

To change model settings:

```python
from src.config import MODEL_CONFIG

# Option 1: Modify the config (affects all subsequent uses)
MODEL_CONFIG["n_estimators"] = 200

# Option 2: Pass parameters directly when creating a model
from src.model import create_model
model = create_model(n_estimators=200)
```

## Common Workflows

### Training a Model

```python
from src import data_loader, feature_engineering, model

# 1. Load and prepare data
train_series, train_events = data_loader.load_train_data(
    series_path, events_path
)
merged = data_loader.merge_series_and_events(train_series, train_events)
merged = feature_engineering.prepare_features(merged, is_training=True)

# 2. Train and evaluate
trained_model, results = model.train_and_evaluate_pipeline(
    merged, verbose=True
)

print(f"Final accuracy: {results['accuracy']:.4f}")
```

### Making Predictions

```python
from src import data_loader, feature_engineering, model, utils
from src.config import EVENT_LABEL_MAP

# 1. Load and prepare test data
test_series = data_loader.load_test_data(test_path)
test_series = feature_engineering.prepare_features(
    test_series, is_training=False
)

# 2. Make predictions
test_features = model.prepare_features_for_model(test_series)
predictions, confidence = model.predict_with_confidence(
    trained_model, test_features
)

# 3. Add predictions to DataFrame
test_series = utils.add_predictions_to_dataframe(
    test_series,
    predictions,
    confidence,
    label_map=EVENT_LABEL_MAP
)

# 4. Export results (optional)
utils.export_submission(test_series, "submission.csv")
```

## Key Benefits

The refactored code provides:

- **Modularity**: Organized into logical, reusable components
- **Flexibility**: Easy to customize individual steps
- **Maintainability**: Clear structure and comprehensive documentation
- **Testability**: Functions can be tested independently
- **Readability**: Clear naming and type hints throughout

## Need Help?

- Check the docstrings in each module for detailed function documentation
- Review the example notebook for a complete workflow
- Run the integration tests to verify your setup
- See the main README.md for more details

## Migration from Original Notebooks

If you're coming from the original notebooks:

1. The original notebooks (`ML - Sleep State Detection.ipynb` and `randomforestclassifier.ipynb`) are kept for reference but have deprecation notices
2. All functionality has been extracted into the `src/` modules
3. The refactored notebook demonstrates the new approach
4. Benefits include better code organization, reusability, and documentation

## Next Steps

1. Update data paths in the refactored notebook or your own scripts
2. Explore the `src/` modules to understand available functions
3. Customize configuration in `src/config.py` as needed
4. Run the complete pipeline on your data
5. Extend or modify modules for your specific use case
