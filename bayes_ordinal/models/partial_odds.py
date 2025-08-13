"""
Partial Proportional Odds Ordinal Regression Model.

This module implements the partial proportional odds ordinal regression model,
which allows some predictors to have proportional effects (same across categories)
while others have category-specific effects (different for each category).

This implementation mimics the functionality of the 'blrm' function from the
'rmsb' R package, providing a flexible approach to ordinal regression.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import arviz as az


def partial_odds_model(y, X, K, proportional_features=None, nonproportional_features=None,
                      link="logit", priors=None, model_name="partial_odds_model", 
                      feature_names=None, group_idx=None, n_groups=None):
    """
    Build a partial proportional odds ordinal regression model.
    
    This model allows:
    - Proportional effects: Some predictors have the same coefficient across all categories
    - Category-specific effects: Other predictors have different coefficients for each category
    
    This mimics the 'blrm' function from the 'rmsb' R package.
    
    Parameters:
    -----------
    y : array-like
        Ordinal response variable (should start from 0)
    X : array-like
        Feature matrix
    K : int
        Number of ordinal categories
    proportional_features : list or None, optional
        Indices or names of features that have proportional effects (same across categories)
        If None, all features are treated as proportional
    nonproportional_features : list or None, optional
        Indices or names of features that have category-specific effects
        If None, no features have category-specific effects
    link : str, optional
        Link function: "logit" or "probit"
    priors : dict, optional
        Prior specifications (if None, uses sensible defaults)
    model_name : str, optional
        Name for the model
    feature_names : list, optional
        Names of features for coefficient naming
    group_idx : array-like, optional
        Group indices for hierarchical modeling
    n_groups : int, optional
        Number of groups for hierarchical modeling
    
    Returns:
    --------
    pm.Model
        PyMC model object with the following structure:
        - beta_proportional: Coefficients for proportional features (size=n_proportional_features)
        - beta_nonproportional: Category-specific coefficients (size=(K-1, n_nonproportional_features))
        - cutpoints: Ordered cutpoints (size=K-1)
        - y: Observed ordinal response
        
    Notes:
    ------
    The model implements:
    - Proportional odds for features in proportional_features
    - Category-specific effects for features in nonproportional_features
    - Proper cumulative probability structure: P(Y ≤ k | X)
    - Support for both logit and probit link functions
    - Hierarchical structure when group_idx and n_groups are provided
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
    
    # Process feature specifications
    n_features = X.shape[1]
    if proportional_features is None and nonproportional_features is None:
        # Default: all features are proportional
        proportional_features = list(range(n_features))
        nonproportional_features = []
    elif proportional_features is None:
        proportional_features = []
    elif nonproportional_features is None:
        nonproportional_features = []
    
    # Convert feature names to indices if needed
    if feature_names is not None:
        if isinstance(proportional_features[0], str):
            proportional_features = [feature_names.index(f) for f in proportional_features]
        if isinstance(nonproportional_features[0], str):
            nonproportional_features = [feature_names.index(f) for f in nonproportional_features]
    
    # Ensure no overlap between proportional and nonproportional features
    if set(proportional_features) & set(nonproportional_features):
        raise ValueError("Features cannot be both proportional and nonproportional")
    
    # Ensure all features are specified
    all_specified = set(proportional_features) | set(nonproportional_features)
    if all_specified != set(range(n_features)):
        raise ValueError(f"All features must be specified. Missing: {set(range(n_features)) - all_specified}")
    
    n_proportional = len(proportional_features)
    n_nonproportional = len(nonproportional_features)
    n_transitions = K - 1
    
    print(f"Model configuration:")
    print(f"  - Proportional features: {n_proportional} (indices: {proportional_features})")
    print(f"  - Non-proportional features: {n_nonproportional} (indices: {nonproportional_features})")
    print(f"  - Total transitions: {n_transitions}")
    
    # Set default priors if none provided
    if priors is None:
        priors = {
            "beta_proportional": [0, 1],      # [mu, sigma] for proportional coefficients
            "beta_nonproportional": [0, 1],   # [mu, sigma] for non-proportional coefficients
            "cutpoints": [0, 2],              # [mu, sigma] for cutpoints
            "u_sigma": 1.0,                   # Group-level variation (hierarchical)
        }
    
    # Ensure y starts from 0
    y_data = np.array(y) - np.min(y)
    
    with pm.Model(name=model_name) as model:
        # 1. Proportional coefficients (same across all categories)
        if n_proportional > 0:
            beta_proportional = pm.Normal(
                "beta_proportional",
                mu=priors.get("beta_proportional", [0, 1])[0],
                sigma=priors.get("beta_proportional", [0, 1])[1],
                shape=n_proportional
            )
            
            # Linear predictor for proportional features
            X_proportional = X[:, proportional_features]
            eta_proportional = pm.math.dot(X_proportional, beta_proportional)
        else:
            eta_proportional = 0
        
        # 2. Non-proportional coefficients (different for each category)
        if n_nonproportional > 0:
            beta_nonproportional = pm.Normal(
                "beta_nonproportional",
                mu=priors.get("beta_nonproportional", [0, 1])[0],
                sigma=priors.get("beta_nonproportional", [0, 1])[1],
                shape=(n_transitions, n_nonproportional)
            )
            
            # Linear predictor for non-proportional features
            X_nonproportional = X[:, nonproportional_features]
            # Shape: (n_samples, n_transitions)
            eta_nonproportional = pm.math.dot(X_nonproportional, beta_nonproportional.T)
        else:
            eta_nonproportional = 0
        
        # 3. Cutpoints (intercepts for each transition)
        cutpoints = pm.Normal(
            "cutpoints",
            mu=priors.get("cutpoints", [0, 2])[0],
            sigma=priors.get("cutpoints", [0, 2])[1],
            transform=pm.distributions.transforms.ordered,
            shape=n_transitions
        )
        
        # 4. Add hierarchical structure if specified
        if group_idx is not None and n_groups is not None:
            u_sigma = priors.get("u_sigma", 1.0)
            u = pm.Normal("u", mu=0, sigma=u_sigma, shape=(n_groups, n_transitions))
            eta_nonproportional = eta_nonproportional + u[group_idx]
        
        # 5. Combine all components for the linear predictor
        # For each transition k: eta_k = cutpoint_k - eta_proportional - eta_nonproportional_k
        # Shape: (n_samples, n_transitions)
        eta = cutpoints[None, :] - eta_proportional[:, None] - eta_nonproportional
        
        # 6. Apply link function to get cumulative probabilities P(Y ≤ k | X)
        if link.lower() == "logit":
            # Logit link: P(Y ≤ k | X) = 1 / (1 + exp(-eta_k))
            cumulative_probs = pm.math.sigmoid(eta)
        elif link.lower() == "probit":
            # Probit link: P(Y ≤ k | X) = Φ(eta_k)
            # Using erf: Φ(x) = 0.5 * (1 + erf(x/sqrt(2)))
            cumulative_probs = 0.5 * (1 + pt.erf(eta / pt.sqrt(2)))
        else:
            raise ValueError("link must be 'logit' or 'probit'")
        
        # 7. Calculate category probabilities P(Y = k | X)
        # P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1)
        cat_probs = pt.zeros((len(y_data), K))
        
        # First category: P(Y = 0) = P(Y ≤ 0)
        cat_probs = pt.set_subtensor(cat_probs[:, 0], cumulative_probs[:, 0])
        
        # Middle categories: P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1)
        for k in range(1, K-1):
            cat_probs = pt.set_subtensor(
                cat_probs[:, k],
                cumulative_probs[:, k] - cumulative_probs[:, k-1]
            )
        
        # Last category: P(Y = K-1) = 1 - P(Y ≤ K-2)
        cat_probs = pt.set_subtensor(
            cat_probs[:, K-1],
            1 - cumulative_probs[:, K-2]
        )
        
        # 8. Ensure probabilities are valid
        cat_probs = pt.clip(cat_probs, 1e-10, 1.0)
        cat_probs = cat_probs / pt.sum(cat_probs, axis=1, keepdims=True)
        
        # 9. Likelihood using Categorical distribution
        pm.Categorical("y", p=cat_probs, observed=y_data)
        
        # Store metadata for reference
        if feature_names is not None:
            model.feature_names = feature_names
        model.proportional_features = proportional_features
        model.nonproportional_features = nonproportional_features
        model.n_transitions = n_transitions
    
    return model


def predict_partial_odds(model, trace, X_new, link="logit"):
    """
    Predict probabilities for new data using a fitted partial proportional odds model.
    
    Parameters:
    -----------
    model : pm.Model
        The fitted PyMC model
    trace : az.InferenceData or pm.MultiTrace
        Posterior samples from the model
    X_new : array-like
        New feature matrix for prediction
    link : str, optional
        Link function used in the model ("logit" or "probit")
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'category_probs': Array of shape (n_samples, n_categories) with P(Y = k | X)
        - 'cumulative_probs': Array of shape (n_samples, n_categories-1) with P(Y ≤ k | X)
        - 'expected_values': Expected ordinal values
        - 'predictions': Most likely category for each sample
    """
    # Extract posterior samples
    if hasattr(trace, 'posterior'):
        # ArviZ InferenceData
        posterior = trace.posterior
        beta_prop = posterior.beta_proportional.values
        beta_nonprop = posterior.beta_nonproportional.values
        cutpoints = posterior.cutpoints.values
    else:
        # PyMC MultiTrace
        beta_prop = trace['beta_proportional']
        beta_nonprop = trace['beta_nonprop']
        cutpoints = trace['cutpoints']
    
    n_samples = X_new.shape[0]
    n_chains = beta_prop.shape[0]
    n_draws = beta_prop.shape[1]
    K = cutpoints.shape[-1] + 1
    
    # Initialize arrays for predictions
    all_cat_probs = np.zeros((n_chains, n_draws, n_samples, K))
    all_cum_probs = np.zeros((n_chains, n_draws, n_samples, K-1))
    
    # Get feature indices
    prop_features = getattr(model, 'proportional_features', [])
    nonprop_features = getattr(model, 'nonproportional_features', [])
    
    for chain in range(n_chains):
        for draw in range(n_draws):
            # Extract parameters for this draw
            beta_p = beta_prop[chain, draw]
            beta_np = beta_nonprop[chain, draw]
            cuts = cutpoints[chain, draw]
            
            # Linear predictor for proportional features
            if len(prop_features) > 0:
                eta_prop = np.dot(X_new[:, prop_features], beta_p)
            else:
                eta_prop = 0
            
            # Linear predictor for non-proportional features
            if len(nonprop_features) > 0:
                eta_nonprop = np.dot(X_new[:, nonprop_features], beta_np.T)
            else:
                eta_nonprop = 0
            
            # Combined linear predictor
            eta = cuts[None, :] - eta_prop[:, None] - eta_nonprop
            
            # Apply link function
            if link.lower() == "logit":
                cum_probs = 1 / (1 + np.exp(-eta))
            elif link.lower() == "probit":
                from scipy.stats import norm
                cum_probs = norm.cdf(eta)
            
            # Calculate category probabilities
            cat_probs = np.zeros((n_samples, K))
            cat_probs[:, 0] = cum_probs[:, 0]
            for k in range(1, K-1):
                cat_probs[:, k] = cum_probs[:, k] - cum_probs[:, k-1]
            cat_probs[:, K-1] = 1 - cum_probs[:, K-2]
            
            # Store results
            all_cat_probs[chain, draw] = cat_probs
            all_cum_probs[chain, draw] = cum_probs
    
    # Aggregate across chains and draws
    cat_probs = np.mean(all_cat_probs, axis=(0, 1))
    cum_probs = np.mean(all_cum_probs, axis=(0, 1))
    
    # Calculate expected values and predictions
    expected_values = np.sum(cat_probs * np.arange(K), axis=1)
    predictions = np.argmax(cat_probs, axis=1)
    
    return {
        'category_probs': cat_probs,
        'cumulative_probs': cum_probs,
        'expected_values': expected_values,
        'predictions': predictions
    }


def compare_models(models_dict, traces_dict, ic="waic"):
    """
    Compare multiple partial proportional odds models using information criteria.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model names to PyMC models
    traces_dict : dict
        Dictionary of model names to posterior traces
    ic : str, optional
        Information criterion to use ("waic", "loo", or "dic")
    
    Returns:
    --------
    pd.DataFrame
        Comparison table with information criteria and model details
    """
    results = []
    
    for name, model in models_dict.items():
        trace = traces_dict[name]
        
        if ic.lower() == "waic":
            try:
                ic_value = az.waic(trace, model)
                ic_se = ic_value.waic_se
                ic_value = ic_value.waic
            except:
                ic_value = np.nan
                ic_se = np.nan
        elif ic.lower() == "loo":
            try:
                ic_value = az.loo(trace, model)
                ic_se = ic_value.loo_se
                ic_value = ic_value.loo
            except:
                ic_value = np.nan
                ic_se = np.nan
        else:
            ic_value = np.nan
            ic_se = np.nan
        
        # Count parameters
        n_params = sum([var.size for var in model.free_RVs])
        
        # Get model features
        prop_features = getattr(model, 'proportional_features', [])
        nonprop_features = getattr(model, 'nonprop_features', [])
        
        results.append({
            'Model': name,
            f'{ic.upper()}': ic_value,
            f'{ic.upper()}_SE': ic_se,
            'Parameters': n_params,
            'Proportional_Features': len(prop_features),
            'Nonproportional_Features': len(nonprop_features)
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values(ic.upper())
    df[f'{ic.upper()}_Rank'] = range(1, len(df) + 1)
    
    return df


def plot_coefficients(trace, model, feature_names=None, figsize=(12, 8)):
    """
    Plot coefficient estimates from a fitted partial proportional odds model.
    
    Parameters:
    -----------
    trace : az.InferenceData or pm.MultiTrace
        Posterior samples from the model
    model : pm.Model
        The fitted PyMC model
    feature_names : list, optional
        Names of features for labeling
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with coefficient plots
    """
    import matplotlib.pyplot as plt
    
    # Extract posterior samples
    if hasattr(trace, 'posterior'):
        posterior = trace.posterior
        beta_prop = posterior.beta_proportional.values
        beta_nonprop = posterior.beta_nonproportional.values
        cutpoints = posterior.cutpoints.values
    else:
        beta_prop = trace['beta_proportional']
        beta_nonprop = trace['beta_nonproportional']
        cutpoints = trace['cutpoints']
    
    # Get feature information
    prop_features = getattr(model, 'proportional_features', [])
    nonprop_features = getattr(model, 'nonproportional_features', [])
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(max(len(prop_features), len(nonprop_features)))]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Partial Proportional Odds Model Coefficients', fontsize=16)
    
    # Plot proportional coefficients
    if len(prop_features) > 0:
        ax = axes[0, 0]
        for i, feat_idx in enumerate(prop_features):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
            beta_samples = beta_prop.reshape(-1, beta_prop.shape[-1])[:, i]
            ax.hist(beta_samples, alpha=0.7, label=feat_name, bins=30)
        ax.set_title('Proportional Coefficients')
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Plot non-proportional coefficients
    if len(nonprop_features) > 0:
        ax = axes[0, 1]
        for i, feat_idx in enumerate(nonprop_features):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
            for k in range(beta_nonprop.shape[-2]):
                beta_samples = beta_nonprop.reshape(-1, beta_nonprop.shape[-2], beta_nonprop.shape[-1])[:, k, i]
                ax.hist(beta_samples, alpha=0.7, label=f'{feat_name}_Cat{k+1}', bins=30)
        ax.set_title('Non-Proportional Coefficients')
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Plot cutpoints
    ax = axes[1, 0]
    for k in range(cutpoints.shape[-1]):
        cut_samples = cutpoints.reshape(-1, cutpoints.shape[-1])[:, k]
        ax.hist(cut_samples, alpha=0.7, label=f'Cutpoint_{k+1}', bins=30)
    ax.set_title('Cutpoints')
    ax.set_xlabel('Cutpoint Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Plot coefficient comparison
    ax = axes[1, 1]
    if len(prop_features) > 0 and len(nonprop_features) > 0:
        # Compare proportional vs non-proportional for a feature that appears in both
        common_features = set(prop_features) & set(nonprop_features)
        if common_features:
            feat_idx = list(common_features)[0]
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
            
            # Proportional coefficient
            prop_samples = beta_prop.reshape(-1, beta_prop.shape[-1])[:, prop_features.index(feat_idx)]
            ax.hist(prop_samples, alpha=0.7, label=f'{feat_name}_Proportional', bins=30)
            
            # Non-proportional coefficients (mean across categories)
            nonprop_samples = np.mean(beta_nonprop.reshape(-1, beta_nonprop.shape[-2], beta_nonprop.shape[-1]), axis=1)
            nonprop_samples = nonprop_samples[:, nonprop_features.index(feat_idx)]
            ax.hist(nonprop_samples, alpha=0.7, label=f'{feat_name}_NonProportional_Mean', bins=30)
            
            ax.set_title(f'Comparison: {feat_name}')
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def test_proportionality_assumption(trace, model, feature_names=None, alpha=0.05):
    """
    Test the proportionality assumption for features in a partial proportional odds model.
    
    Parameters:
    -----------
    trace : az.InferenceData or pm.MultiTrace
        Posterior samples from the model
    model : pm.Model
        The fitted PyMC model
    feature_names : list, optional
        Names of features for labeling
    alpha : float, optional
        Significance level for the test
    
    Returns:
    --------
    pd.DataFrame
        Results of proportionality tests for each feature
    """
    # Extract posterior samples
    if hasattr(trace, 'posterior'):
        posterior = trace.posterior
        beta_prop = posterior.beta_proportional.values
        beta_nonprop = posterior.beta_nonproportional.values
    else:
        beta_prop = trace['beta_proportional']
        beta_nonprop = trace['beta_nonproportional']
    
    # Get feature information
    prop_features = getattr(model, 'proportional_features', [])
    nonprop_features = getattr(model, 'nonproportional_features', [])
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(max(len(prop_features), len(nonprop_features)))]
    
    results = []
    
    # Test proportional features
    for i, feat_idx in enumerate(prop_features):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
        
        # For proportional features, we test if the coefficient is significantly different from zero
        beta_samples = beta_prop.reshape(-1, beta_prop.shape[-1])[:, i]
        
        # Calculate credible interval
        lower = np.percentile(beta_samples, alpha/2 * 100)
        upper = np.percentile(beta_samples, (1-alpha/2) * 100)
        
        # Test if zero is in the credible interval
        is_significant = not (lower <= 0 <= upper)
        
        results.append({
            'Feature': feat_name,
            'Type': 'Proportional',
            'Mean': np.mean(beta_samples),
            'Std': np.std(beta_samples),
            'Lower_CI': lower,
            'Upper_CI': upper,
            'Significant': is_significant,
            'Proportionality_Test': 'N/A (Proportional by design)'
        })
    
    # Test non-proportional features
    for i, feat_idx in enumerate(nonprop_features):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
        
        # For non-proportional features, we test if coefficients are significantly different across categories
        beta_samples = beta_nonprop.reshape(-1, beta_nonprop.shape[-2], beta_nonprop.shape[-1])
        feat_coeffs = beta_samples[:, :, i]  # Shape: (chains*draws, categories, feature)
        
        # Calculate mean coefficient across categories
        mean_coeff = np.mean(feat_coeffs, axis=1)
        
        # Test if coefficients vary significantly across categories
        # Using coefficient of variation as a measure
        cv = np.std(feat_coeffs, axis=1) / np.abs(np.mean(feat_coeffs, axis=1))
        cv = np.where(np.abs(np.mean(feat_coeffs, axis=1)) < 1e-10, 0, cv)
        
        # Test if the variation is significant
        cv_threshold = 0.1  # 10% variation threshold
        is_variable = np.mean(cv) > cv_threshold
        
        # Calculate credible intervals for each category
        category_means = np.mean(feat_coeffs, axis=0)
        category_stds = np.std(feat_coeffs, axis=0)
        
        results.append({
            'Feature': feat_name,
            'Type': 'Non-Proportional',
            'Mean': np.mean(mean_coeff),
            'Std': np.std(mean_coeff),
            'Lower_CI': np.percentile(mean_coeff, alpha/2 * 100),
            'Upper_CI': np.percentile(mean_coeff, (1-alpha/2) * 100),
            'Significant': np.any(np.abs(category_means) > 2 * category_stds),
            'Proportionality_Test': f'Variable (CV={np.mean(cv):.3f})' if is_variable else 'Proportional'
        })
    
    return pd.DataFrame(results)


def create_partial_odds_summary(trace, model, feature_names=None):
    """
    Create a comprehensive summary of a fitted partial proportional odds model.
    
    Parameters:
    -----------
    trace : az.InferenceData or pm.MultiTrace
        Posterior samples from the model
    model : pm.Model
        The fitted PyMC model
    feature_names : list, optional
        Names of features for labeling
    
    Returns:
    --------
    dict
        Comprehensive model summary including:
        - Model information
        - Coefficient summaries
        - Model fit statistics
        - Predictions
    """
    # Extract posterior samples
    if hasattr(trace, 'posterior'):
        posterior = trace.posterior
        beta_prop = posterior.beta_proportional.values
        beta_nonprop = posterior.beta_nonproportional.values
        cutpoints = posterior.cutpoints.values
    else:
        beta_prop = trace['beta_proportional']
        beta_nonprop = trace['beta_nonproportional']
        cutpoints = trace['cutpoints']
    
    # Get feature information
    prop_features = getattr(model, 'proportional_features', [])
    nonprop_features = getattr(model, 'nonproportional_features', [])
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(max(len(prop_features), len(nonprop_features)))]
    
    # Model information
    model_info = {
        'Model_Type': 'Partial Proportional Odds',
        'Proportional_Features': len(prop_features),
        'Nonproportional_Features': len(nonprop_features),
        'Total_Features': len(prop_features) + len(nonprop_features),
        'Categories': cutpoints.shape[-1] + 1,
        'Transitions': cutpoints.shape[-1]
    }
    
    # Coefficient summaries
    coef_summary = {}
    
    # Proportional coefficients
    if len(prop_features) > 0:
        for i, feat_idx in enumerate(prop_features):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
            beta_samples = beta_prop.reshape(-1, beta_prop.shape[-1])[:, i]
            
            coef_summary[f'{feat_name}_Proportional'] = {
                'Mean': float(np.mean(beta_samples)),
                'Std': float(np.std(beta_samples)),
                '2.5%': float(np.percentile(beta_samples, 2.5)),
                '50%': float(np.percentile(beta_samples, 50)),
                '97.5%': float(np.percentile(beta_samples, 97.5)),
                'Type': 'Proportional'
            }
    
    # Non-proportional coefficients
    if len(nonprop_features) > 0:
        for i, feat_idx in enumerate(nonprop_features):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
            beta_samples = beta_nonprop.reshape(-1, beta_nonprop.shape[-2], beta_nonprop.shape[-1])
            feat_coeffs = beta_samples[:, :, i]
            
            for k in range(feat_coeffs.shape[1]):
                cat_samples = feat_coeffs[:, k]
                coef_summary[f'{feat_name}_Cat{k+1}'] = {
                    'Mean': float(np.mean(cat_samples)),
                    'Std': float(np.std(cat_samples)),
                    '2.5%': float(np.percentile(cat_samples, 2.5)),
                    '50%': float(np.percentile(cat_samples, 50)),
                    '97.5%': float(np.percentile(cat_samples, 97.5)),
                    'Type': 'Non-Proportional'
                }
    
    # Cutpoint summaries
    cutpoint_summary = {}
    for k in range(cutpoints.shape[-1]):
        cut_samples = cutpoints.reshape(-1, cutpoints.shape[-1])[:, k]
        cutpoint_summary[f'Cutpoint_{k+1}'] = {
            'Mean': float(np.mean(cut_samples)),
            'Std': float(np.std(cut_samples)),
            '2.5%': float(np.percentile(cut_samples, 2.5)),
            '50%': float(np.percentile(cut_samples, 50)),
            '97.5%': float(np.percentile(cut_samples, 97.5))
        }
    
    # Model fit statistics
    try:
        waic = az.waic(trace, model)
        loo = az.loo(trace, model)
        fit_stats = {
            'WAIC': float(waic.waic),
            'WAIC_SE': float(waic.waic_se),
            'LOO': float(loo.loo),
            'LOO_SE': float(loo.loo_se)
        }
    except:
        fit_stats = {'WAIC': np.nan, 'WAIC_SE': np.nan, 'LOO': np.nan, 'LOO_SE': np.nan}
    
    return {
        'Model_Info': model_info,
        'Coefficients': coef_summary,
        'Cutpoints': cutpoint_summary,
        'Fit_Statistics': fit_stats
    }