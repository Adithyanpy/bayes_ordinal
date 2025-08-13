"""
Utility functions for Bayesian ordinal regression.
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
        X_df = X.copy()
        if categorical_cols is None:
            categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        X_df = pd.DataFrame(X)
        if categorical_cols is None:
            categorical_cols = []
    
    encoders = {}
    X_encoded = X_df.copy()
    
    for col in categorical_cols:
        if col in X_df.columns:
            encoder = LabelEncoder()
            X_encoded[col] = encoder.fit_transform(X_df[col].astype(str))
            encoders[col] = encoder
    
    return X_encoded.values, encoders

def standardize_features(
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features for better MCMC sampling.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None, fit a new one.
        
    Returns
    -------
    X_scaled : np.ndarray
        Standardized feature matrix.
    scaler : StandardScaler
        Fitted scaler for later use.
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
    Create group indices for hierarchical models.
    
    Parameters
    ----------
    group_var : array-like
        Group variable (can be strings, integers, etc.)
        
    Returns
    -------
    group_idx : np.ndarray
        Integer indices for groups.
    n_groups : int
        Number of unique groups.
    group_map : dict
        Mapping from original values to indices.
    """
    group_var = np.asarray(group_var)
    unique_groups = np.unique(group_var)
    n_groups = len(unique_groups)
    
    # Create mapping
    group_map = {group: idx for idx, group in enumerate(unique_groups)}
    
    # Create indices
    group_idx = np.array([group_map[g] for g in group_var])
    
    return group_idx, n_groups, group_map

def compute_category_proportions(y: np.ndarray, K: int) -> np.ndarray:
    """
    Compute proportions of each category.
    
    Parameters
    ----------
    y : np.ndarray
        Ordinal response variable.
    K : int
        Number of categories.
        
    Returns
    -------
    proportions : np.ndarray
        Array of proportions for each category.
    """
    counts = np.bincount(y, minlength=K)
    return counts / counts.sum()

def check_convergence(
    idata,
    var_names: Optional[list] = None,
    rhat_threshold: float = 1.1,
    ess_threshold: float = 400
) -> Dict[str, Any]:
    """
    Check MCMC convergence diagnostics.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from PyMC.
    var_names : list, optional
        Variables to check.
    rhat_threshold : float
        Maximum acceptable R-hat value.
    ess_threshold : float
        Minimum acceptable effective sample size.
        
    Returns
    -------
    diagnostics : dict
        Dictionary with convergence diagnostics.
    """
    import arviz as az
    
    # Check if we have enough samples for reliable diagnostics
    n_chains = idata.posterior.dims.get('chain', 0)
    n_draws = idata.posterior.dims.get('draw', 0)
    
    if n_chains < 2 or n_draws < 4:
        print(f"Warning: Insufficient samples for reliable diagnostics (chains={n_chains}, draws={n_draws})")
        print("ArviZ requires at least 2 chains and 4 draws for accurate diagnostics")
        return {
            'n_bad_rhat': 0,
            'n_bad_ess': 0,
            'n_divergences': 0,
            'rhat_max': float('nan'),
            'ess_min': float('nan'),
            'converged': False,
            'summary': pd.DataFrame(),
            'insufficient_samples': True
        }
    
    try:
        summary = az.summary(idata, var_names=var_names)
        
        # Check R-hat
        rhat_ok = summary['r_hat'] < rhat_threshold
        n_bad_rhat = (~rhat_ok).sum()
        
        # Check ESS
        ess_ok = summary['ess_bulk'] > ess_threshold
        n_bad_ess = (~ess_ok).sum()
        
        # Check divergences
        n_divergences = idata.sample_stats['diverging'].sum().item()
        
        # Extract max R-hat and min ESS for compatibility
        rhat_max = summary['r_hat'].max() if len(summary) > 0 else float('nan')
        ess_min = summary['ess_bulk'].min() if len(summary) > 0 else float('nan')
        
        return {
            'n_bad_rhat': n_bad_rhat,
            'n_bad_ess': n_bad_ess,
            'n_divergences': n_divergences,
            'rhat_max': rhat_max,
            'ess_min': ess_min,
            'converged': (n_bad_rhat == 0) and (n_bad_ess == 0) and (n_divergences == 0),
            'summary': summary,
            'insufficient_samples': False
        }
    except Exception as e:
        print(f"Error in convergence diagnostics: {e}")
        return {
            'n_bad_rhat': 0,
            'n_bad_ess': 0,
            'n_divergences': 0,
            'rhat_max': float('nan'),
            'ess_min': float('nan'),
            'converged': False,
            'summary': pd.DataFrame(),
            'error': str(e)
        }

def validate_ordinal_model(y: np.ndarray, X: np.ndarray, K: int) -> dict:
    """
    Validate ordinal regression model setup according to PyMC best practices.
    
    Parameters
    ----------
    y : np.ndarray
        Ordinal response variable.
    X : np.ndarray
        Covariate matrix.
    K : int
        Number of ordinal categories.
        
    Returns
    -------
    dict
        Validation results and recommendations.
        
    References
    ----------
    PyMC Documentation: https://docs.pymc.io/en/latest/learn/core_notebooks/ordinal_regression.html
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "recommendations": []
    }
    
    # Check data dimensions
    if len(y) != X.shape[0]:
        validation_results["valid"] = False
        validation_results["warnings"].append("Response and predictor dimensions don't match")
    
    # Check ordinal response (PyMC requirements)
    unique_y = np.unique(y)
    if len(unique_y) != K:
        validation_results["warnings"].append(f"Expected {K} categories, found {len(unique_y)}")
    
    if not np.all(np.diff(unique_y) == 1):
        validation_results["warnings"].append("Response categories are not consecutive integers")
    
    # Check for sparse categories (PyMC recommendation)
    category_counts = np.bincount(y, minlength=K)
    min_count = category_counts.min()
    if min_count < 5:
        validation_results["warnings"].append(f"Category {np.argmin(category_counts)} has only {min_count} observations")
        validation_results["recommendations"].append("Consider combining sparse categories")
    
    # Check predictor scaling (PyMC best practice)
    X_std = X.std(axis=0)
    if np.any(X_std < 0.1):
        validation_results["warnings"].append("Some predictors have very low variance")
        validation_results["recommendations"].append("Consider standardizing predictors")
    
    # Check for multicollinearity (PyMC recommendation)
    if X.shape[1] > 1:
        try:
            corr_matrix = np.corrcoef(X.T)
            high_corr = np.abs(corr_matrix - np.eye(corr_matrix.shape[0])) > 0.8
            if np.any(high_corr):
                validation_results["warnings"].append("High correlation detected between predictors")
                validation_results["recommendations"].append("Consider removing correlated predictors")
        except Exception as e:
            validation_results["warnings"].append(f"Could not check multicollinearity: {e}")
    
    # Sample size recommendations (PyMC guidelines)
    n_samples = len(y)
    if n_samples < 50:
        validation_results["warnings"].append("Small sample size may lead to unstable estimates")
        validation_results["recommendations"].append("Consider using more conservative priors")
    
    if n_samples < 10 * X.shape[1]:
        validation_results["warnings"].append("Sample size may be insufficient for number of predictors")
        validation_results["recommendations"].append("Consider reducing model complexity")
    
    # PyMC-specific checks
    if n_samples < 100:
        validation_results["recommendations"].append("Consider using more chains and longer sampling for small datasets")
    
    if X.shape[1] > 20:
        validation_results["warnings"].append("High-dimensional model may require specialized sampling")
        validation_results["recommendations"].append("Consider using NUTS with target_accept=0.95")
    
    return validation_results
