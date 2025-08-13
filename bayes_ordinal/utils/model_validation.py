"""
Model validation utilities for Bayesian ordinal regression.
"""

import numpy as np
import arviz as az
from typing import Dict, Any, Optional


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
        Inference data from PyMC sampling.
    var_names : list, optional
        Variable names to check. If None, checks all variables.
    rhat_threshold : float
        Threshold for R-hat statistic.
    ess_threshold : float
        Threshold for effective sample size.
        
    Returns
    -------
    dict
        Dictionary containing convergence diagnostics.
    """
    diagnostics = {
        "converged": True,
        "rhat_issues": [],
        "ess_issues": [],
        "summary": {}
    }
    
    # Get summary statistics
    summary = az.summary(idata, var_names=var_names)
    
    # Check R-hat
    rhat_vars = summary[summary['r_hat'] > rhat_threshold]
    if not rhat_vars.empty:
        diagnostics["converged"] = False
        diagnostics["rhat_issues"] = rhat_vars.index.tolist()
    
    # Check ESS
    ess_vars = summary[summary['ess_bulk'] < ess_threshold]
    if not ess_vars.empty:
        diagnostics["converged"] = False
        diagnostics["ess_issues"] = ess_vars.index.tolist()
    
    # Store summary
    diagnostics["summary"] = summary
    
    return diagnostics


def validate_ordinal_model(y: np.ndarray, X: np.ndarray, K: int) -> dict:
    """
    Validate ordinal regression model inputs.
    
    Parameters
    ----------
    y : np.ndarray
        Response variable.
    X : np.ndarray
        Feature matrix.
    K : int
        Number of categories.
        
    Returns
    -------
    dict
        Validation results.
    """
    validation = {
        "valid": True,
        "issues": [],
        "warnings": []
    }
    
    # Check data types
    if not isinstance(y, np.ndarray):
        validation["issues"].append("y must be a numpy array")
        validation["valid"] = False
    
    if not isinstance(X, np.ndarray):
        validation["issues"].append("X must be a numpy array")
        validation["valid"] = False
    
    if not isinstance(K, int) or K < 2:
        validation["issues"].append("K must be an integer >= 2")
        validation["valid"] = False
    
    # Check shapes
    if len(y.shape) != 1:
        validation["issues"].append("y must be 1-dimensional")
        validation["valid"] = False
    
    if len(X.shape) != 2:
        validation["issues"].append("X must be 2-dimensional")
        validation["valid"] = False
    
    # Check sample sizes
    if len(y) != X.shape[0]:
        validation["issues"].append("y and X must have the same number of samples")
        validation["valid"] = False
    
    # Check y values
    y_min, y_max = y.min(), y.max()
    if y_min < 0:
        validation["issues"].append("y contains negative values")
        validation["valid"] = False
    
    if y_max >= K:
        validation["issues"].append(f"y contains values >= K (max={y_max}, K={K})")
        validation["valid"] = False
    
    # Check for gaps in categories
    unique_vals = np.unique(y)
    expected_vals = np.arange(K)
    if not np.array_equal(unique_vals, expected_vals):
        validation["warnings"].append(f"y contains gaps: found {unique_vals}, expected {expected_vals}")
    
    # Check feature matrix
    if np.any(np.isnan(X)):
        validation["issues"].append("X contains missing values")
        validation["valid"] = False
    
    if np.any(np.isinf(X)):
        validation["issues"].append("X contains infinite values")
        validation["valid"] = False
    
    return validation
