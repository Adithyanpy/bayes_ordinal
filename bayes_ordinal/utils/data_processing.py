"""
Data processing utilities for Bayesian ordinal regression.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder


def validate_ordinal_data(
    y: Union[np.ndarray, pd.Series, list],
    X: Union[np.ndarray, pd.DataFrame],
    K: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Validate and preprocess ordinal regression data.
    
    Parameters
    ----------
    y : array-like
        Ordinal response variable.
    X : array-like
        Feature matrix.
    K : int, optional
        Number of categories. If None, inferred from y.
        
    Returns
    -------
    y_clean : np.ndarray
        Cleaned and validated response variable.
    X_clean : np.ndarray
        Cleaned and validated feature matrix.
    K : int
        Number of categories.
        
    Raises
    ------
    ValueError
        If data validation fails.
    """
    # Convert to numpy arrays
    y = np.asarray(y).ravel()
    
    # Handle different input types for X
    if isinstance(X, pd.DataFrame):
        X_clean = X.copy()
        if X_clean.isnull().any().any():
            raise ValueError("Feature matrix contains missing values")
    else:
        X_clean = np.asarray(X)
        if np.any(np.isnan(X_clean)):
            raise ValueError("Feature matrix contains missing values")
    
    # Check shapes
    if len(y) != X_clean.shape[0]:
        raise ValueError(f"y has {len(y)} samples but X has {X_clean.shape[0]} samples")
    
    # Check for missing values in y
    if np.any(np.isnan(y)):
        raise ValueError("Response variable contains missing values")
    
    # Validate y values
    y_min, y_max = y.min(), y.max()
    if y_min < 0:
        raise ValueError(f"y contains negative values (min={y_min})")
    
    # Determine K
    if K is None:
        K = int(y_max) + 1
    else:
        if y_max >= K:
            raise ValueError(f"y contains values >= K (max={y_max}, K={K})")
    
    # Check for gaps in categories
    unique_vals = np.unique(y)
    expected_vals = np.arange(K)
    if not np.array_equal(unique_vals, expected_vals):
        raise ValueError(f"y contains gaps: found {unique_vals}, expected {expected_vals}")
    
    return y, X_clean, K


def encode_categorical_features(
    X: Union[np.ndarray, pd.DataFrame],
    categorical_cols: Optional[list] = None
) -> Tuple[np.ndarray, Dict[str, LabelEncoder]]:
    """
    Encode categorical features for ordinal regression.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    categorical_cols : list, optional
        Column indices or names for categorical features.
        
    Returns
    -------
    X_encoded : np.ndarray
        Encoded feature matrix.
    encoders : dict
        Dictionary mapping column names to LabelEncoder objects.
    """
    if isinstance(X, pd.DataFrame):
        X_clean = X.copy()
        feature_names = X_clean.columns.tolist()
    else:
        X_clean = pd.DataFrame(X)
        feature_names = [f"feature_{i}" for i in range(X_clean.shape[1])]
    
    # Auto-detect categorical columns if not specified
    if categorical_cols is None:
        categorical_cols = []
        for col in feature_names:
            if X_clean[col].dtype == 'object' or X_clean[col].nunique() < 10:
                categorical_cols.append(col)
    
    encoders = {}
    
    for col in categorical_cols:
        if col in feature_names:
            le = LabelEncoder()
            X_clean[col] = le.fit_transform(X_clean[col].astype(str))
            encoders[col] = le
    
    return X_clean.values, encoders


def standardize_features(
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features for ordinal regression.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None, a new one is created and fitted.
        
    Returns
    -------
    X_scaled : np.ndarray
        Standardized feature matrix.
    scaler : StandardScaler
        Fitted scaler object.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def create_group_indices(
    group_var: Union[np.ndarray, pd.Series, list]
) -> Tuple[np.ndarray, int, Dict[Any, int]]:
    """
    Create group indices for hierarchical modeling.
    
    Parameters
    ----------
    group_var : array-like
        Group variable (e.g., subject IDs, site IDs).
        
    Returns
    -------
    group_idx : np.ndarray
        Zero-based group indices.
    n_groups : int
        Number of unique groups.
    group_mapping : dict
        Mapping from original values to indices.
    """
    group_var = np.asarray(group_var)
    unique_groups = np.unique(group_var)
    n_groups = len(unique_groups)
    
    # Create mapping
    group_mapping = {group: idx for idx, group in enumerate(unique_groups)}
    
    # Create indices
    group_idx = np.array([group_mapping[g] for g in group_var])
    
    return group_idx, n_groups, group_mapping


def compute_category_proportions(y: np.ndarray, K: int) -> np.ndarray:
    """
    Compute proportions of each category.
    
    Parameters
    ----------
    y : np.ndarray
        Response variable.
    K : int
        Number of categories.
        
    Returns
    -------
    proportions : np.ndarray
        Array of proportions for each category.
    """
    counts = np.bincount(y, minlength=K)
    return counts / len(y)
