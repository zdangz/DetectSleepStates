# DetectSleepStates

A machine learning project for detecting sleep onset and wakeup events from accelerometer data, developed for the [Child Mind Institute - Detect Sleep States competition on Kaggle](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states).

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Approach](#model-approach)
- [Project Structure](#project-structure)
- [Results](#results)
- [Development Notes](#development-notes)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project uses machine learning techniques to automatically detect sleep onset and wakeup events from wrist-worn accelerometer data. The goal is to improve sleep analysis for children and adolescents, particularly those with mood and behavior difficulties, by providing accurate automated sleep event detection.

## 📖 Problem Description

The project aims to detect two types of sleep events from continuous accelerometer data:
- **Sleep Onset**: The moment when a person transitions from wakefulness to sleep
- **Wakeup**: The moment when a person transitions from sleep to wakefulness

Traditional sleep studies require manual annotation by experts, which is time-consuming and expensive. This automated approach can make sleep analysis more accessible and scalable.

## 📊 Dataset

The dataset consists of accelerometer data collected from wrist-worn devices:

### Training Data

- **train_series.parquet**: Time-series accelerometer data
  - `series_id`: Unique identifier for each participant
  - `step`: Time step (5-second intervals)
  - `timestamp`: ISO 8601 formatted timestamp
  - `anglez`: Z-angle of the accelerometer (measures arm angle relative to vertical)
  - `enmo`: Euclidean Norm Minus One (measure of movement intensity)

- **train_events.csv**: Labeled sleep events
  - `series_id`: Participant identifier
  - `night`: Night number for the participant
  - `event`: Type of event (`onset` or `wakeup`)
  - `step`: Time step where the event occurred
  - `timestamp`: Event timestamp

### Data Characteristics

- Over 127 million rows of accelerometer readings
- Data collected at 5-second intervals
- Contains both labeled events and unlabeled time periods
- Some missing values in the events dataset for incomplete recordings

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:
```bash
git clone https://github.com/zdangz/DetectSleepStates.git
cd DetectSleepStates
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn jupyter pyarrow
```

3. Download the competition data from Kaggle:
   - Visit the [competition data page](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data)
   - Download `train_series.parquet` and `train_events.csv`
   - Place them in your working directory or update the file paths in the notebooks

## 💻 Usage

### Running the Notebooks

1. **Main Analysis Notebook** (`ML - Sleep State Detection.ipynb`):
   - Open in Jupyter: `jupyter notebook "ML - Sleep State Detection.ipynb"`
   - Update file paths in the first cell to point to your data location
   - Run cells sequentially to:
     - Load and explore the data
     - Clean and preprocess the datasets
     - Perform feature engineering
     - Train the Random Forest model
     - Make predictions on test data

2. **Kaggle Submission Notebook** (`randomforestclassifier.ipynb`):
   - Designed to run on Kaggle's platform
   - Uses Kaggle's input paths (`/kaggle/input/...`)
   - Generates submission file in the required format

### Key Steps

1. **Data Import**: Load the parquet and CSV files
2. **Data Cleansing**: Handle missing values and data quality issues
3. **Data Merging**: Combine series and events data on matching keys
4. **Feature Engineering**: 
   - Extract time-based features (hour, day of week)
   - Calculate rolling averages for anglez and enmo
5. **Model Training**: Train Random Forest classifier
6. **Prediction**: Generate predictions for test data

## 🤖 Model Approach

### Features

The model uses the following engineered features:

1. **Time Features**:
   - Hour of the day
   - Day of the week
   
2. **Accelerometer Features**:
   - Rolling average of Z-angle (window size: 10 steps = 50 seconds)
   - Rolling average of ENMO (Euclidean Norm Minus One)

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - `n_estimators`: 100 trees
  - `random_state`: 42 (for reproducibility)
- **Target**: Binary classification (onset: 0, wakeup: 1)

### Training Process

1. Merge time-series data with labeled events
2. Engineer time-based and rolling window features
3. Split data into training (80%) and validation (20%) sets
4. Train Random Forest model on training set
5. Evaluate on validation set
6. Generate predictions for test data

## 📁 Project Structure

```
DetectSleepStates/
├── ML - Sleep State Detection.ipynb    # Main analysis and model development
├── randomforestclassifier.ipynb        # Kaggle submission notebook
├── submission.csv                       # Sample submission file
├── README.md                            # This file
├── DEVELOPER_GUIDE.md                   # Detailed developer documentation
└── CONTRIBUTING.md                      # Contribution guidelines
```

## 📈 Results

The model achieves reasonable validation accuracy for sleep event detection. The submission file contains predictions with confidence scores for each time step and event type combination.

### Output Format

The submission CSV contains:
- `row_id`: Unique identifier for each prediction
- `series_id`: Participant identifier
- `step`: Time step
- `event`: Predicted event type (`onset` or `wakeup`)
- `score`: Confidence score (0-1)

## 🔧 Development Notes

### Version History

**28 October 2023**:
- Fixed "Notebook out of memory" issue during submission by:
  - Deleting unused dataframes
  - Importing and using the `gc` (garbage collection) module
- Debugging "Submission scoring error"

### Known Issues

- Large dataset requires significant memory (127M+ rows)
- Some participants have missing event labels
- Model performance may vary by participant due to individual differences

### Memory Optimization Tips

1. Delete unused dataframes immediately after use
2. Use garbage collection (`gc.collect()`) regularly
3. Process data in chunks when possible
4. Use appropriate dtypes to reduce memory footprint

## 📚 References

### Helpful Resources

- **Exploratory Data Analysis**:
  - [Accelerometer Data Analysis by dumisanisibanda](https://www.kaggle.com/code/dumisanisibanda/exploratory-accelerometer-data-analysis)
  - [Sleep EDA Plots by yihsuankao](https://www.kaggle.com/code/yihsuankao/sleep-eda-plots)

- **Model Baselines**:
  - [Random Forest Model Starter by sumitai](https://www.kaggle.com/code/sumitai/zzzs-random-forest-model-starter)

### Competition Information

- [Child Mind Institute - Detect Sleep States Competition](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)
- [Competition Discussion Forum](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

For detailed development information, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

## 📄 License

This project is part of a Kaggle competition. Please refer to the [competition rules](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/rules) for data usage and submission guidelines.

## 👥 Authors

- [@zdangz](https://github.com/zdangz)

## 🙏 Acknowledgments

- Child Mind Institute for organizing the competition
- Kaggle community for shared notebooks and insights
- All contributors who provided exploratory analysis and baseline models
