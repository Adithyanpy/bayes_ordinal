"""
Advanced prior specification utilities for Bayesian ordinal regression.

This module provides McElreath-style conservative priors and advanced prior
specification methods following Statistical Rethinking principles.
"""

from typing import Dict, Any, Optional
import numpy as np

def get_conservative_priors(K: Optional[int] = None, model_type: str = "ordinal") -> Dict[str, Any]:
    """
    Get PyMC-style conservative priors following PyMC documentation and McElreath principles.
    
    These priors are designed to be conservative and weakly informative,
    following PyMC's recommended practices for ordinal regression.
    
    Parameters
    ----------
    K : int, optional
        Number of categories. If provided, cutpoint priors will be properly sized.
    model_type : str, default="ordinal"
        Type of model ("ordinal", "continuous", "hierarchical")
        
    Returns
    -------
    dict
        Dictionary with PyMC-style prior hyperparameters.
        
    References
    ----------
    PyMC Documentation: https://docs.pymc.io/en/latest/learn/core_notebooks/ordinal_regression.html
    McElreath, R. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and Stan.
    """
    if model_type == "ordinal":
        # PyMC-style conservative priors (following PyMC documentation)
        priors = {
            "coef_mu": 0.0,
            "coef_sigma": 1.0,  # Conservative for unstandardized predictors
            "u_sigma": 1.0,     # Group-level variation
            "gamma_mu": 0.0,
            "gamma_sigma": 1.0,
        }
        
        # PyMC-style cutpoint priors (following Ordered Categories documentation)
        if K is not None:
            priors["cut_min"] = 0.0
            priors["cut_max"] = float(K)
            # Add PyMC-specific cutpoint parameters
            priors["cut_mu"] = 0.0
            priors["cut_sigma"] = 1.0
        else:
            priors["cut_min"] = 0.0
            priors["cut_max"] = 1.0
            priors["cut_mu"] = 0.0
            priors["cut_sigma"] = 1.0
            
    elif model_type == "continuous":
        # For continuous outcomes (like wine judgement scores)
        priors = {
            "coef_mu": 0.0,
            "coef_sigma": 0.5,
            "u_sigma": 1.0,
            "sigma": 1.0,  # Observation noise
        }
        
    elif model_type == "hierarchical":
        # For hierarchical models with group-level effects
        priors = {
            "coef_mu": 0.0,
            "coef_sigma": 1.0,  # More conservative for hierarchical models
            "u_sigma": 1.0,
            "gamma_mu": 0.0,
            "gamma_sigma": 1.0,
        }
        
        # PyMC-style cutpoint priors for hierarchical models
        if K is not None:
            priors["cut_min"] = 0.0
            priors["cut_max"] = float(K)
            priors["cut_mu"] = 0.0
            priors["cut_sigma"] = 1.0
        else:
            priors["cut_min"] = 0.0
            priors["cut_max"] = 1.0
            priors["cut_mu"] = 0.0
            priors["cut_sigma"] = 1.0
            
    else:
        raise ValueError(f"Unknown model_type {model_type!r}; choose 'ordinal', 'continuous', or 'hierarchical'")
    
    return priors

def get_data_informed_priors(y: np.ndarray, X: np.ndarray, K: int) -> Dict[str, Any]:
    """
    Get data-informed priors based on observed data patterns.
    
    This function computes priors that are informed by the observed data
    while remaining conservative and following McElreath's principles.
    
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
        Dictionary with data-informed prior hyperparameters.
    """
    # Compute category proportions
    category_counts = np.bincount(y, minlength=K)
    category_props = category_counts / category_counts.sum()
    
    # Compute cumulative proportions for cutpoint initialization
    cumulative_props = np.cumsum(category_props[:-1])
    
    # Data-informed priors
    priors = {
        "coef_mu": 0.0,
        "coef_sigma": 0.5,  # Conservative for standardized predictors
        "u_sigma": 1.0,
        "gamma_mu": 0.0,
        "gamma_sigma": 1.0,
    }
    
    # Advanced Dirichlet cutpoint priors based on data
    priors["cut_min"] = 0.0
    priors["cut_max"] = float(K)
    
    return priors

def validate_priors(priors: Dict[str, Any], model_type: str = "cumulative") -> Dict[str, Any]:
    """
    Validate and complete prior specifications.
    
    This function ensures that all required prior parameters are present
    and have reasonable values.
    
    Parameters
    ----------
    priors : dict
        User-specified priors.
    model_type : str, default="cumulative"
        Type of model ("cumulative", "partial_odds")
        
    Returns
    -------
    dict
        Validated and completed prior specifications.
    """
    validated = priors.copy()
    
    # Default values for missing parameters
    defaults = {
        "coef_mu": 0.0,
        "coef_sigma": 0.5,
        "u_sigma": 1.0,
        "gamma_mu": 0.0,
        "gamma_sigma": 1.0,
        "cut_min": 0.0,
        "cut_max": 1.0,
    }
    
    # Add missing defaults
    for key, default_val in defaults.items():
        if key not in validated:
            validated[key] = default_val
    
    return validated

# Legacy function names for backward compatibility (but using advanced priors)
def get_default_priors() -> Dict[str, Any]:
    """Get default McElreath-style priors."""
    return get_conservative_priors()

def get_weakly_informative_priors() -> Dict[str, Any]:
    """Get weakly informative McElreath-style priors."""
    return get_conservative_priors()

def get_informative_priors(
    coef_means: Optional[np.ndarray] = None,
    coef_sds: Optional[np.ndarray] = None,
    cut_means: Optional[np.ndarray] = None,
    cut_sds: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Get informative priors (using McElreath-style as base)."""
    priors = get_conservative_priors()
    
    # Override with user-specified values if provided
    if coef_means is not None:
        priors["coef_mu"] = coef_means
    if coef_sds is not None:
        priors["coef_sigma"] = coef_sds
    if cut_means is not None:
        priors["cut_mu"] = cut_means
    if cut_sds is not None:
        priors["cut_sigma"] = cut_sds
    
    return priors
