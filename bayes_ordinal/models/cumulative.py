"""
Cumulative (Proportional Odds) Ordinal Regression Model.

This module implements the cumulative ordinal regression model with support
for both constrained and flexible cutpoint approaches, and hierarchical modeling.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from typing import Dict, Any, Optional


def cumulative_model(y, X, K, link="logit", priors=None, model_name="cumulative_model", 
                    feature_names=None, group_idx=None, n_groups=None, prior_type="fixed_sigma", 
                    auto_probit_adjustment=True):
    """
    Build a cumulative ordinal regression model.
    
    Parameters:
    -----------
    y : array-like
        Ordinal response variable (should start from 0)
    X : array-like
        Feature matrix
    K : int
        Number of ordinal categories
    link : str, optional
        Link function: "logit" or "probit"
    priors : dict, optional
        Prior specifications (if None, uses sensible defaults). Key parameters:
        - "beta": [mu, sigma] for coefficients (default: [0, 1])
        - "sigma": sigma for cutpoints (default: 1.0)
        - "mu": means for cutpoints (default: zeros for fixed_sigma, linspace for exponential_sigma)
        - "tau": scale parameter for HalfNormal prior on group-level variation (default: 1.0)
        - "constrained_uniform": whether to use constrained Dirichlet approach (default: False)
    model_name : str, optional
        Name for the model
    feature_names : list, optional
        Names of features for coefficient naming
    group_idx : array-like, optional
        Group indices for hierarchical modeling
    n_groups : int, optional
        Number of groups for hierarchical modeling
    prior_type : str, optional
        "fixed_sigma": Fixed sigma for cutpoints
        "exponential_sigma": Random sigma from Exponential prior
    auto_probit_adjustment : bool, optional
        Whether to automatically adjust prior scales for probit models (default: True).
        Set to False when using data-informed priors like z-scores that are already
        properly scaled for the probit link function.
    
    Returns:
    --------
    pm.Model
        PyMC model object with the following structure:
        - beta: Generic coefficients (size=n_features)
        - cutpoints: Ordered cutpoints (size=K-1)
        - y: Observed ordinal response
        - eta: Linear predictor
        - u: Group-level effects (if hierarchical)
        - u_sigma: Group-level variation parameter (if hierarchical)
        
    Notes:
    ------
    The model features:
    - Generic beta coefficients with size parameter
    - Support for both constrained (Dirichlet) and flexible (Normal) cutpoint approaches
    - Hierarchical structure when group_idx and n_groups are provided
    - Automatic 0-based indexing for ordinal categories
    - Two prior approaches: fixed_sigma (fixed sigma) and exponential_sigma (random sigma)
    - Automatic probit adjustment: Prior scales are automatically adjusted for probit models
      (coefficients, cutpoints, and group-level variation are scaled by ~0.625).
      This can be disabled with auto_probit_adjustment=False when using properly scaled priors.
    - Additional model attributes: feature_names, link (accessible after model creation)
    """
    # Validate inputs
    if K < 2:
        raise ValueError("K must be at least 2 for ordinal regression")
    
    if len(y) != X.shape[0]:
        raise ValueError(f"Length of y ({len(y)}) must match number of rows in X ({X.shape[0]})")
    
    # Check if X is 2D
    if len(X.shape) != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    
    # Check if X has at least one feature
    if X.shape[1] < 1:
        raise ValueError(f"X must have at least one feature, got {X.shape[1]} features")
    
    # Validate feature_names if provided
    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError(f"feature_names length ({len(feature_names)}) must match number of features ({X.shape[1]})")
    
    # Validate group parameters
    if (group_idx is not None) != (n_groups is not None):
        raise ValueError("Both group_idx and n_groups must be provided together for hierarchical modeling")
    
    if group_idx is not None and len(group_idx) != len(y):
        raise ValueError("group_idx length must match number of samples")
    
    # Validate prior_type
    if prior_type not in ["fixed_sigma", "exponential_sigma"]:
        raise ValueError("prior_type must be 'fixed_sigma' or 'exponential_sigma'")
    
    # Set default priors if none provided
    if priors is None:
        if prior_type == "fixed_sigma":
            priors = {
                "beta": [0, 1],       # [mu, sigma] for coefficients
                "sigma": 1.0,         # Fixed sigma for cutpoints
                "mu": np.zeros(K-1),  # mu=0 for all cutpoints
                "tau": 1.0,           # Scale parameter for HalfNormal prior on u_sigma
                "constrained_uniform": False  # Use flexible approach by default
            }
        else:  # exponential_sigma
            priors = {
                "beta": [0, 1],       # [mu, sigma] for coefficients
                "sigma": 1.0,         # Exponential prior parameter
                "mu": np.linspace(0, K, K-1),  # Default cutpoint means
                "tau": 1.0,           # Scale parameter for HalfNormal prior on u_sigma
                "constrained_uniform": False  # Use flexible approach by default
            }
    
    # Apply probit adjustment factor if using probit link and auto-adjustment is enabled
    # Probit coefficients are approximately 1.6 times smaller than logit coefficients
    # This adjustment provides more appropriate prior scales for probit models
    # Note: Disable this when using data-informed priors (e.g., z-scores) that are already
    # properly scaled for the probit link function
    if link.lower() == "probit" and auto_probit_adjustment:
        probit_adjustment = 1.0 / 1.6  # ~ 0.625
        
        # Adjust coefficient priors
        if "beta" in priors and len(priors["beta"]) == 2:
            priors["beta"][1] *= probit_adjustment  # Adjust sigma only
        
        # Adjust cutpoint priors
        if "sigma" in priors:
            priors["sigma"] *= probit_adjustment
        
        # Adjust group-level variation scale parameter
        if "tau" in priors:
            priors["tau"] *= probit_adjustment
    
    # Ensure y starts from 0
    y_data = np.array(y) - np.min(y)
    
    with pm.Model(name=model_name) as model:
        # Add coordinates for better ArviZ output
        coords = {
            "obs": np.arange(len(y_data)),
            "feature": feature_names or [f"x{i}" for i in range(X.shape[1])],
            "cut": np.arange(K-1),
        }
        if group_idx is not None and n_groups is not None:
            coords["group"] = np.arange(n_groups)
        
        model.add_coords(coords)
        
        # Generic coefficient structure with dims
        n_features = X.shape[1]
        beta = pm.Normal("beta", 
                         mu=priors.get("beta", [0, 1])[0], 
                         sigma=priors.get("beta", [0, 1])[1], 
                         dims=("feature",))
        
        # Create pm.Data variable for the entire feature matrix
        X_data = pm.Data("X", X, dims=("obs", "feature"))
        
        # Linear predictor using pm.Data variable
        eta_base = pm.math.dot(X_data, beta)
        
        # Add hierarchical structure if specified
        if group_idx is not None and n_groups is not None:
            tau = priors.get("tau", 1.0)
            u_sigma = pm.HalfNormal("u_sigma", sigma=tau)
            u = pm.Normal("u", mu=0, sigma=u_sigma, dims=("group",))
            eta = pm.Deterministic("eta", eta_base + u[group_idx], dims=("obs",))
        else:
            eta = pm.Deterministic("eta", eta_base, dims=("obs",))
        
        # Cutpoints implementation
        constrained_uniform = priors.get("constrained_uniform", False)
        
        if constrained_uniform:
            # Approach 1: Constrained Dirichlet approach (standard method)
            # This ensures proper ordering and constraint properties for ordinal cutpoints
            # The Dirichlet distribution ensures proportions sum to 1, creating ordered cutpoints
            cuts_unknown = pm.Dirichlet("cuts_unknown", a=np.ones(K-2))
            cutpoints = pm.Deterministic(
                "cutpoints",
                pt.concatenate([
                    np.zeros(1),  # First cutpoint at 0
                    pt.cumsum(cuts_unknown)  # Cumulative sums for remaining cutpoints
                ])
            )
        else:
            if prior_type == "fixed_sigma":
                # Approach 2a: Fixed sigma
                cutpoints = pm.Normal(
                    "cutpoints",
                    mu=priors.get("mu", np.zeros(K-1)),
                    sigma=priors.get("sigma", 1.0),  # Fixed sigma value
                    transform=pm.distributions.transforms.ordered,
                    dims=("cut",)
                )
            else:
                # Approach 2b: Exponential sigma
                sigma = pm.Exponential("sigma", priors.get("sigma", 1.0))
                cutpoints = pm.Normal(
                    "cutpoints",
                    mu=priors.get("mu", np.linspace(0, K, K-1)),
                    sigma=sigma,  # Random sigma from Exponential
                    transform=pm.distributions.transforms.ordered,
                    dims=("cut",)
                )
        
        # Likelihood
        if link.lower() == "logit":
            pm.OrderedLogistic("y", cutpoints=cutpoints, eta=eta, observed=y_data, dims=("obs",))
        elif link.lower() == "probit":
            pm.OrderedProbit("y", cutpoints=cutpoints, eta=eta, observed=y_data, dims=("obs",))
        else:
            raise ValueError("link must be 'logit' or 'probit'")
        
        # Store feature names for reference if provided
        if feature_names is not None:
            model.feature_names = feature_names
        
        # Store link function for other modules to access
        model.link = link.lower()
    
    return model
