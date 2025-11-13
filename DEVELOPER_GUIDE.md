# Developer Guide

This guide provides detailed technical information for developers working on the DetectSleepStates project.

## 📋 Table of Contents

- [Project Architecture](#project-architecture)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Implementation](#model-implementation)
- [Code Organization](#code-organization)
- [Development Workflow](#development-workflow)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

## 🏗️ Project Architecture

### Overview

The project follows a standard machine learning pipeline architecture:

```
Data Loading → Data Cleaning → Feature Engineering → Model Training → Prediction → Submission
```

### Technology Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Data Storage**: Parquet format (for large time-series data)
- **Development Environment**: Jupyter Notebook

### Design Principles

1. **Modularity**: Each stage of the pipeline is independent
2. **Reproducibility**: Fixed random seeds for consistent results
3. **Memory Efficiency**: Explicit memory management for large datasets
4. **Incremental Development**: Step-by-step validation of each component

## 📊 Data Pipeline

### 1. Data Loading

#### train_series.parquet
```python
train_series = pd.read_parquet(path1)
```

**Why Parquet?**
- Columnar storage format optimized for analytics
- Better compression than CSV (smaller file size)
- Faster read times for large datasets
- Preserves data types automatically

**Schema**:
- `series_id` (string): Participant identifier
- `step` (int64): Sequential time step
- `timestamp` (datetime): ISO 8601 timestamp with timezone
- `anglez` (float64): Z-angle measurement
- `enmo` (float64): Movement intensity metric

#### train_events.csv
```python
train_events = pd.read_csv(path2)
```

**Schema**:
- `series_id` (string): Matches train_series
- `night` (int64): Sleep session number
- `event` (string): "onset" or "wakeup"
- `step` (float64): Time step (NaN for missing events)
- `timestamp` (datetime): Event timestamp (NaN for missing events)

### 2. Data Cleansing

#### Missing Value Analysis

```python
# Check for NaN values
nan_rows_train_series = train_series[train_series.isna().any(axis=1)]
nan_rows_train_events = train_events[train_events.isna().any(axis=1)]
```

**Key Findings**:
- train_series: 0 missing values (clean sensor data)
- train_events: ~4,923 rows with NaN (incomplete event labels)

#### Cleaning Strategy

```python
train_events_cleaned = train_events.dropna()
```

**Rationale**:
- Missing events indicate incomplete recordings or unlabeled periods
- Cannot use for supervised learning without labels
- Preserves ~9,587 valid event labels (66% retention)

**Alternative Approaches** (Not Implemented):
- Imputation based on typical sleep patterns
- Semi-supervised learning using unlabeled data
- Multi-task learning with missing label indicators

### 3. Data Merging

```python
merged_data = pd.merge(train_series, train_events, 
                       on=['series_id', 'step', 'timestamp'])
```

**Join Type**: Inner join
- Only keeps rows where events are labeled
- Results in ~9,587 labeled time points from 127M+ sensor readings
- Creates highly imbalanced dataset (99.99% of data unlabeled)

**Memory Impact**:
- Input: 127M rows × 5 columns ≈ 5GB RAM
- Output: ~9,587 rows × 8 columns ≈ 1MB RAM
- Significant reduction enables in-memory processing

## 🔧 Feature Engineering

### Time Features

#### Timestamp Parsing

```python
# Remove timezone for consistent parsing
merged_data['timestamp_notimezone'] = merged_data['timestamp'].str[:-5]

# Convert to datetime with UTC
merged_data['timestamp_utc'] = pd.to_datetime(
    merged_data['timestamp_notimezone'], 
    utc=True
)
```

**Timezone Handling**:
- Original data has various timezones (-0400, -0500, etc.)
- Strip timezone suffix for consistent datetime parsing
- Convert to UTC for standardization
- Preserves relative time relationships

#### Time-Based Features

```python
# Extract hour (0-23)
merged_data['hour'] = merged_data['timestamp_utc'].dt.hour

# Extract day name (Monday, Tuesday, etc.)
merged_data['day_of_week'] = merged_data['timestamp_utc'].dt.day_name()
```

**Feature Rationale**:

1. **Hour of Day**:
   - Sleep patterns are strongly circadian
   - Most sleep onset: 21:00-23:00
   - Most wakeup: 06:00-08:00
   - Provides strong discriminative signal

2. **Day of Week**:
   - Weekend vs weekday sleep patterns differ
   - Potential social jetlag effects
   - Not currently used in model but available for future work

### Rolling Window Features

#### Window Size Selection

```python
window_size = 10  # 10 steps = 50 seconds
```

**Design Considerations**:
- Step interval: 5 seconds
- Window: 10 steps = 50 seconds
- Captures short-term movement patterns
- Smooths sensor noise
- Small enough to detect rapid changes

#### Rolling Average Implementation

```python
# Sort by time to ensure proper window calculation
merged_data = merged_data.sort_values(by='time')

# Calculate rolling averages
merged_data['anglez_rolling_avg'] = merged_data['anglez'].rolling(
    window=window_size, 
    min_periods=1
).mean()

merged_data['enmo_rolling_avg'] = merged_data['enmo'].rolling(
    window=window_size, 
    min_periods=1
).mean()
```

**Parameters**:
- `window`: Number of observations in the window
- `min_periods=1`: Allows calculation for early rows with incomplete windows
- `mean()`: Simple moving average

**Feature Interpretation**:

1. **anglez_rolling_avg**:
   - Smoothed arm angle
   - Sleep: typically stable, near horizontal
   - Wake: more variable, changing positions
   - Onset: transition to stable horizontal
   - Wakeup: transition to active movement

2. **enmo_rolling_avg**:
   - Smoothed movement intensity
   - Sleep: near-zero (minimal movement)
   - Wake: elevated (active movement)
   - Provides complementary signal to angle

### Label Encoding

```python
merged_data['event_label'] = merged_data['event'].map({
    'onset': 0, 
    'wakeup': 1
})
```

**Binary Classification**:
- 0: Sleep onset event
- 1: Wakeup event
- Maps categorical labels to numeric for sklearn

## 🤖 Model Implementation

### Model Selection

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
```

**Why Random Forest?**

**Advantages**:
- Handles non-linear relationships naturally
- Robust to outliers and noisy data
- No feature scaling required
- Built-in feature importance
- Resistant to overfitting with proper tuning
- Fast training on moderate-sized datasets

**Alternative Models** (Not Implemented):
- Gradient Boosting (XGBoost, LightGBM): Often better performance
- Neural Networks (LSTM, CNN): Better for sequence modeling
- Logistic Regression: Simpler baseline
- Support Vector Machines: Potential for better boundaries

**Hyperparameters**:
- `n_estimators=100`: Number of decision trees
  - More trees → better performance, slower training
  - 100 is reasonable default for initial model
- `random_state=42`: Ensures reproducibility
  - Same results across runs
  - Important for debugging and comparison

### Feature Selection

```python
X = merged_data[['hour', 'anglez_rolling_avg', 'enmo_rolling_avg']]
y = merged_data['event_label']
```

**Selected Features**:
1. `hour`: Time of day (0-23)
2. `anglez_rolling_avg`: Smoothed arm angle
3. `enmo_rolling_avg`: Smoothed movement intensity

**Excluded Features**:
- `series_id`: Identifier, not predictive
- `step`: Arbitrary sequence number
- `timestamp`: Redundant with `hour`
- `day_of_week`: Low predictive value for binary classification
- `night`: Session number, not directly predictive

### Training Process

```python
from sklearn.model_selection import train_test_split

# Split data: 80% training, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Train model
model.fit(X_train, y_train)
```

**Train/Validation Split**:
- **80/20 split**: Standard ratio for moderate datasets
- **random_state=42**: Reproducible splits
- **No stratification**: Classes reasonably balanced (~50/50 onset/wakeup)

**Important Considerations**:
- Split is random, not by participant
- May have data leakage if same participant in train and validation
- Better approach: split by `series_id` for true generalization

### Model Evaluation

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
```

**Evaluation Metrics**:

1. **Accuracy** (Currently Used):
   - Simple, interpretable
   - Appropriate for balanced classes
   - May not capture all aspects of performance

2. **Additional Metrics** (Recommended):
   ```python
   from sklearn.metrics import classification_report, confusion_matrix
   
   print(classification_report(y_val, y_pred))
   print(confusion_matrix(y_val, y_pred))
   ```
   - Precision: How many predicted events are correct?
   - Recall: How many actual events are detected?
   - F1-score: Harmonic mean of precision and recall
   - Confusion Matrix: Shows specific error patterns

### Prediction for Submission

```python
# Predict probabilities
y_pred_proba = model.predict_proba(X_test)

# Create submission dataframe
submission = pd.DataFrame({
    'row_id': row_ids,
    'series_id': series_ids,
    'step': steps,
    'event': events,
    'score': y_pred_proba[:, 1]  # Probability of positive class
})
```

**Output Format**:
- `score`: Confidence score (0-1) for each event prediction
- Higher score = more confident prediction
- Allows for threshold tuning in post-processing

## 📂 Code Organization

### Notebook Structure

#### ML - Sleep State Detection.ipynb

**Section 1: Data Import**
- Load train_series.parquet
- Load train_events.csv
- Display basic statistics

**Section 2: Data Cleansing**
- Check for missing values
- Remove incomplete records
- Validate data quality

**Section 3: Data Merging**
- Join series and events on keys
- Verify merge results

**Section 4: Feature Engineering**
- 4.1 Time Information
  - Parse timestamps
  - Extract hour and day features
- 4.2 Sliding Window
  - Sort by time
  - Calculate rolling averages

**Section 5: Model Selection**
- Initialize Random Forest
- Encode target labels

**Section 6: Training**
- Split train/validation
- Fit model
- Evaluate accuracy

**Section 7: Classification on Test Data**
- Load test data
- Apply same transformations
- Generate predictions
- Create submission file

#### randomforestclassifier.ipynb

**Purpose**: Kaggle submission version
- Uses Kaggle-specific input paths
- Optimized for Kaggle kernel environment
- Memory-efficient for submission scoring

**Key Differences**:
- Input paths: `/kaggle/input/...`
- Memory management: Explicit `gc.collect()`
- Submission generation: Direct CSV output

### Best Practices

1. **Cell Organization**:
   - One logical operation per cell
   - Clear markdown headers
   - Descriptive variable names

2. **Code Style**:
   - Follow PEP 8 conventions
   - Use meaningful variable names
   - Add comments for complex logic

3. **Output Management**:
   - Print shape after each transformation
   - Display head() to verify changes
   - Show descriptive statistics

## 🔄 Development Workflow

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn jupyter pyarrow

# Start Jupyter
jupyter notebook
```

### Typical Development Cycle

1. **Data Exploration**:
   ```python
   # Check data shape
   print(df.shape)
   
   # View sample data
   print(df.head())
   
   # Check data types
   print(df.dtypes)
   
   # Look for missing values
   print(df.isnull().sum())
   ```

2. **Feature Development**:
   ```python
   # Create new feature
   df['new_feature'] = ...
   
   # Validate feature
   print(df['new_feature'].describe())
   
   # Check for NaN
   print(df['new_feature'].isna().sum())
   ```

3. **Model Training**:
   ```python
   # Train model
   model.fit(X_train, y_train)
   
   # Evaluate
   score = model.score(X_val, y_val)
   print(f"Validation Score: {score}")
   ```

4. **Iteration**:
   - Analyze errors
   - Refine features
   - Tune hyperparameters
   - Re-train and evaluate

### Version Control

```bash
# Check status
git status

# Stage changes
git add notebook.ipynb

# Commit
git commit -m "Descriptive message"

# Push
git push origin main
```

**Notebook Versioning Tips**:
- Clear all outputs before committing: `Cell → All Output → Clear`
- Use meaningful commit messages
- Tag important versions
- Consider using nbdime for notebook diffs

## ⚡ Performance Optimization

### Memory Management

#### Problem: Large Dataset Memory Usage

The train_series.parquet file contains 127M+ rows, requiring ~5GB RAM.

#### Solutions:

1. **Early Filtering**:
   ```python
   # Only keep relevant series_id values
   relevant_ids = train_events['series_id'].unique()
   train_series = train_series[train_series['series_id'].isin(relevant_ids)]
   ```

2. **Chunked Processing**:
   ```python
   # Process in chunks
   chunk_size = 1_000_000
   for chunk in pd.read_parquet(path, chunksize=chunk_size):
       # Process chunk
       process_chunk(chunk)
   ```

3. **Data Type Optimization**:
   ```python
   # Use smaller data types
   df['series_id'] = df['series_id'].astype('category')
   df['step'] = df['step'].astype('int32')
   df['anglez'] = df['anglez'].astype('float32')
   ```

4. **Explicit Garbage Collection**:
   ```python
   import gc
   
   # Delete unused dataframes
   del train_series
   del train_events
   
   # Force garbage collection
   gc.collect()
   ```

### Computational Performance

#### Vectorization

Replace loops with vectorized operations:

```python
# Slow: Loop
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'old_col'] * 2

# Fast: Vectorized
df['new_col'] = df['old_col'] * 2
```

#### Efficient Aggregations

```python
# Use groupby for aggregations
grouped_stats = df.groupby('series_id').agg({
    'anglez': ['mean', 'std'],
    'enmo': ['mean', 'std']
})
```

#### Rolling Windows

Already vectorized with pandas:
```python
# Efficient built-in method
df['rolling_mean'] = df['value'].rolling(window=10).mean()
```

### Model Training Performance

#### Parallel Processing

```python
# Use all CPU cores
model = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,  # Use all available cores
    random_state=42
)
```

#### Feature Matrix Optimization

```python
# Convert to numpy array for faster training
X_array = X.to_numpy()
model.fit(X_array, y)
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors

**Error**: `MemoryError` or kernel crash

**Solutions**:
- Reduce chunk size
- Process data in batches
- Use data type optimization
- Close unnecessary applications
- Increase system swap/page file

#### 2. File Path Issues

**Error**: `FileNotFoundError`

**Solutions**:
```python
import os

# Check if file exists
if not os.path.exists(path):
    print(f"File not found: {path}")
    
# Use absolute paths
path = os.path.abspath('train_series.parquet')

# Print current directory
print(f"Current directory: {os.getcwd()}")
```

#### 3. Missing Values in Rolling Window

**Error**: NaN in first rows after rolling calculation

**Expected Behavior**:
- First rows have incomplete windows
- Using `min_periods=1` allows calculation
- Alternative: Drop first N rows after calculation

#### 4. Submission Format Errors

**Error**: Kaggle submission scoring error

**Validation**:
```python
# Check submission format
print(submission.head())
print(submission.columns)
print(submission.shape)

# Verify no missing values
print(submission.isnull().sum())

# Check data types
print(submission.dtypes)

# Verify row_id is unique
print(submission['row_id'].nunique() == len(submission))
```

#### 5. Model Training Issues

**Error**: Poor performance or no convergence

**Debugging Steps**:
```python
# Check feature distributions
X.describe()

# Look for NaN in features
X.isnull().sum()

# Check label distribution
y.value_counts()

# Verify train/test split
print(f"Train: {len(X_train)}, Test: {len(X_val)}")
```

### Debugging Techniques

#### 1. Data Inspection

```python
# Display DataFrame info
df.info()

# Show sample rows
df.sample(10)

# Check unique values
df['column'].value_counts()
```

#### 2. Intermediate Results

```python
# Save intermediate results
df.to_parquet('intermediate_data.parquet')

# Load for debugging
debug_df = pd.read_parquet('intermediate_data.parquet')
```

#### 3. Profiling

```python
import time

start = time.time()
# ... code to profile ...
end = time.time()
print(f"Execution time: {end - start:.2f} seconds")
```

## 🚀 Future Improvements

### Model Enhancements

#### 1. Advanced Features

**Frequency Domain Features**:
```python
from scipy import signal

# FFT for frequency analysis
fft = np.fft.fft(df['anglez'].values)
power_spectrum = np.abs(fft)**2
```

**Statistical Features**:
```python
# Per-window statistics
df['anglez_std'] = df['anglez'].rolling(window=10).std()
df['enmo_max'] = df['enmo'].rolling(window=10).max()
df['enmo_min'] = df['enmo'].rolling(window=10).min()
```

**Activity Recognition**:
```python
# Movement intensity categories
df['activity_level'] = pd.cut(
    df['enmo'], 
    bins=[0, 0.01, 0.05, float('inf')],
    labels=['still', 'light', 'active']
)
```

#### 2. Better Models

**Gradient Boosting**:
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

**Deep Learning**:
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(sequence_length, n_features)),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
```

#### 3. Sequence Modeling

**Problem**: Current model treats each time step independently

**Solution**: Use sequence models to capture temporal patterns

```python
# Create sequences
def create_sequences(df, sequence_length=60):
    sequences = []
    labels = []
    for i in range(len(df) - sequence_length):
        seq = df.iloc[i:i+sequence_length][['anglez', 'enmo']].values
        label = df.iloc[i+sequence_length]['event_label']
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)
```

### Data Improvements

#### 1. Better Train/Val Split

```python
# Split by participant to prevent leakage
unique_ids = merged_data['series_id'].unique()
train_ids, val_ids = train_test_split(
    unique_ids, 
    test_size=0.2, 
    random_state=42
)

train_data = merged_data[merged_data['series_id'].isin(train_ids)]
val_data = merged_data[merged_data['series_id'].isin(val_ids)]
```

#### 2. Data Augmentation

```python
# Add noise for robustness
def augment_data(df, noise_level=0.01):
    df_aug = df.copy()
    df_aug['anglez'] += np.random.normal(0, noise_level, len(df))
    df_aug['enmo'] += np.random.normal(0, noise_level, len(df))
    return df_aug
```

#### 3. Handle Class Imbalance

```python
from imblearn.over_sampling import SMOTE

# Oversample minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### Pipeline Improvements

#### 1. Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

pipeline.fit(X_train, y_train)
```

#### 2. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

#### 3. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, 
    cv=5, 
    scoring='accuracy'
)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Production Readiness

#### 1. Model Serialization

```python
import joblib

# Save model
joblib.dump(model, 'sleep_detection_model.pkl')

# Load model
loaded_model = joblib.load('sleep_detection_model.pkl')
```

#### 2. API Endpoint

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = extract_features(data)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})
```

#### 3. Monitoring

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log predictions
logger.info(f"Prediction: {prediction}, Confidence: {confidence}")
```

## 📚 Additional Resources

### Documentation
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [NumPy Documentation](https://numpy.org/doc/)

### Tutorials
- [Kaggle Learn: Machine Learning](https://www.kaggle.com/learn/machine-learning)
- [Time Series with Python](https://www.kaggle.com/learn/time-series)

### Papers
- Random Forests: Breiman, L. (2001). "Random Forests". Machine Learning.
- Sleep Analysis: Research papers on actigraphy and sleep detection

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

For general project information, see [README.md](README.md).

---

**Last Updated**: November 2024  
**Maintainer**: [@zdangz](https://github.com/zdangz)
