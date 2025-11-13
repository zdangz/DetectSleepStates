"""
Model training and prediction module for Sleep State Detection.

This module handles model creation, training, evaluation, and making
predictions on new data.
"""

from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from .config import MODEL_CONFIG, TRAIN_TEST_SPLIT_CONFIG, FEATURE_COLUMNS


def create_model(
    n_estimators: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs
) -> RandomForestClassifier:
    """
    Create a Random Forest Classifier model.
    
    Args:
        n_estimators: Number of trees in the forest. If None, uses config value.
        random_state: Random state for reproducibility. If None, uses config value.
        **kwargs: Additional parameters to pass to RandomForestClassifier
        
    Returns:
        Initialized RandomForestClassifier instance
    """
    if n_estimators is None:
        n_estimators = MODEL_CONFIG["n_estimators"]
    if random_state is None:
        random_state = MODEL_CONFIG["random_state"]
    
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        **kwargs
    )


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and validation sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data to use for validation. If None, uses config value.
        random_state: Random state for reproducibility. If None, uses config value.
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    if test_size is None:
        test_size = TRAIN_TEST_SPLIT_CONFIG["test_size"]
    if random_state is None:
        random_state = TRAIN_TEST_SPLIT_CONFIG["random_state"]
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )


def train_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train the model on training data.
    
    Args:
        model: RandomForestClassifier instance
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model performance on validation data.
    
    Args:
        model: Trained RandomForestClassifier
        X_val: Validation features
        y_val: True validation labels
        verbose: If True, print evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    results = {
        "accuracy": accuracy,
        "predictions": y_pred,
        "true_labels": y_val
    }
    
    if verbose:
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
    
    return results


def prepare_features_for_model(
    df: pd.DataFrame,
    feature_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Extract feature columns needed for model prediction.
    
    Args:
        df: DataFrame containing all columns
        feature_columns: List of feature column names. If None, uses config value.
        
    Returns:
        DataFrame containing only feature columns
        
    Raises:
        KeyError: If any required feature column is missing
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required feature columns: {missing_cols}")
    
    return df[feature_columns]


def predict_with_confidence(
    model: RandomForestClassifier,
    X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with confidence scores.
    
    Args:
        model: Trained RandomForestClassifier
        X: Features to predict on
        
    Returns:
        Tuple of (predicted_labels, confidence_scores)
        - predicted_labels: Array of predicted class labels
        - confidence_scores: Array of confidence scores (max probability)
    """
    # Get predicted class probabilities
    class_probabilities = model.predict_proba(X)
    
    # Get the predicted labels (the class with the highest probability)
    predicted_labels = model.classes_[class_probabilities.argmax(axis=1)]
    
    # Get the confidence scores (the highest class probability for each sample)
    confidence_scores = class_probabilities.max(axis=1)
    
    return predicted_labels, confidence_scores


def train_and_evaluate_pipeline(
    df: pd.DataFrame,
    feature_columns: Optional[list] = None,
    label_column: str = 'event_label',
    test_size: Optional[float] = None,
    model_params: Optional[dict] = None,
    verbose: bool = True
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Complete training and evaluation pipeline.
    
    This convenience function performs all steps: feature preparation,
    train/validation split, model creation, training, and evaluation.
    
    Args:
        df: DataFrame with features and labels
        feature_columns: List of feature column names. If None, uses config value.
        label_column: Name of the label column
        test_size: Proportion for validation set. If None, uses config value.
        model_params: Additional parameters for model creation
        verbose: If True, print progress and results
        
    Returns:
        Tuple of (trained_model, evaluation_results)
        
    Raises:
        KeyError: If required columns are missing
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in DataFrame")
    
    # Prepare features and labels
    X = prepare_features_for_model(df, feature_columns)
    y = df[label_column]
    
    if verbose:
        print(f"Training with {len(feature_columns)} features on {len(df)} samples")
        print(f"Features: {', '.join(feature_columns)}")
    
    # Split data
    X_train, X_val, y_train, y_val = split_train_validation(
        X, y, test_size=test_size
    )
    
    if verbose:
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
    
    # Create and train model
    model_params = model_params or {}
    model = create_model(**model_params)
    
    if verbose:
        print("\nTraining model...")
    model = train_model(model, X_train, y_train)
    
    # Evaluate
    if verbose:
        print("\nEvaluating model...")
    results = evaluate_model(model, X_val, y_val, verbose=verbose)
    
    return model, results
