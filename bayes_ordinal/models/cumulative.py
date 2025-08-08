"""
Cumulative (Proportional Odds) Ordinal Regression Model - FIXED VERSION.

This module implements the cumulative ordinal regression model following
PyMC Ordered Categories (OC) documentation and McElreath's Statistical 
Rethinking principles. It provides exact implementations matching the
OC documentation structure and parameterization.

FIXED VERSION: Enables log-likelihood computation for model comparison by removing
problematic initvals and using informative priors for numerical stability.

Key Features:
- Follows PyMC Ordered Categories documentation exactly
- Uses pm.Data() for flexible predictions
- Uses "response" variable name (not "y")
- Uses phi as linear predictor name
- FIXED: Uses informative priors instead of initvals for cutpoints
- Uses pm.OrderedLogistic() and pm.OrderedProbit() as in OC documentation
- Supports both logit and probit link functions
- Includes counterfactual analysis following OC documentation
- Enables full model comparison workflow (LOO/WAIC)

References:
- PyMC Ordered Categories Documentation
- McElreath's Statistical Rethinking
"""

import numpy as np
import pymc as pm
import pandas as pd
import pytensor.tensor as pt
import xarray as xr
from typing import Dict, Any, Optional, Tuple, Union


def cumulative_model(
    y: Union[np.ndarray, pd.Series],
    X: Union[np.ndarray, pd.DataFrame],
    K: Optional[int] = None,
    link: str = "logit",
    priors: Optional[Dict[str, Any]] = None,
    model_name: str = "cumulative_model",
    feature_names: Optional[list] = None,
    group_idx: Optional[np.ndarray] = None,
    n_groups: Optional[int] = None
) -> pm.Model:
    """
    Build cumulative ordinal regression model following PyMC Ordered Categories documentation.
    
    This function creates a model that matches the PyMC Ordered Categories (OC) documentation
    structure exactly, using pm.Data() for inputs and pm.OrderedLogistic()/pm.OrderedProbit()
    for the likelihood. It follows the exact parameterization from the OC documentation.
    
    FIXED VERSION: Removes initvals from cutpoints to enable log-likelihood computation
    while maintaining numerical stability through informative priors.
    
    Key Features:
    - Exact PyMC OC documentation implementation
    - Uses pm.Data() for all features (enables counterfactual analysis)
    - Uses "response" as observed variable name (OC documentation standard)
    - Uses phi as linear predictor name (following OC)
    - FIXED: Informative priors on cutpoints instead of initvals
    - Supports hierarchical intercepts with non-centered parameterization
    - Enables full Bayesian workflow including model comparison
    - Compatible with all existing workflow functions
    
    Parameters
    ----------
    y : array-like of shape (n,)
        Ordinal response variable (will be converted to 0-based indexing)
    X : array-like of shape (n, p)
        Covariate matrix (features)
    K : int, optional
        Number of ordinal categories. If None, inferred from unique values in y.
    link : {"logit", "probit"}, default="logit"
        Link function for the cumulative model
    priors : dict, optional
        Dictionary of prior hyperparameters with keys:
            - "coef_mu", "coef_sigma" : mean and std for coefficients
            - "cut_mu", "cut_sigma"   : mean and std for cutpoints
            - "u_sigma"               : std for group intercepts (hierarchical)
        Defaults use conservative values for stability.
    model_name : str, default="cumulative_model"
        Name for the PyMC model object
    feature_names : list, optional
        Names for features. If None, uses X column names or creates generic names.
    group_idx : array-like, optional
        Group indices for hierarchical modeling (0-based integer codes)
    n_groups : int, optional
        Number of groups for hierarchical modeling
        
    Returns
    -------
    pm.Model
        PyMC model following OC documentation structure with optional hierarchical intercepts
        that supports log-likelihood computation for model comparison.
        
    Notes
    -----
    This fixed version:
    - Removes initvals from cutpoints to enable log-likelihood computation
    - Uses informative prior means on cutpoints for numerical stability
    - Maintains exact API compatibility with original implementation
    - Supports all existing workflow functions and counterfactual analysis
    - Enables full Bayesian model comparison workflow
        
    References
    ----------
    PyMC Ordered Categories Documentation: 
    https://docs.pymc.io/en/latest/learn/core_notebooks/ordinal_regression.html
    
    Examples
    --------
    >>> model = cumulative_model(y, X, link="logit", feature_names=["x1", "x2"])
    >>> with model:
    ...     idata = pm.sample(idata_kwargs={"log_likelihood": True})
    >>> loo = az.loo(idata)  # Now works without conflicts!
    """
    
    # Handle DataFrame input
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X.columns)
        X_array = X.values
    else:
        X_array = X
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
    
    # Handle Series input
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    # Infer K if not provided
    if K is None:
        K = len(np.unique(y_array))
    
    # Factorize response following PyMC documentation
    RESPONSE_ID, RESPONSE = pd.factorize(y_array, sort=True)
    N_RESPONSE_CLASSES = len(RESPONSE)
    CUTPOINTS = np.arange(1, N_RESPONSE_CLASSES).astype(int)
    
    # Set up coordinates for cutpoints
    coords = {"CUTPOINTS": CUTPOINTS}
    
    # Use conservative default priors if none provided (FIXED VERSION)
    if priors is None:
        priors = {
            "coef_mu": 0.0,
            "coef_sigma": 0.5,    # Conservative (following PyMC documentation)
            "cut_mu": 0.0,
            "cut_sigma": 1.5,     # Slightly wider for stability without initvals
            "u_sigma": 1.0        # Group-level variation (hierarchical)
        }
    
    with pm.Model(coords=coords, name=model_name) as model:
        
        # Data for posterior predictions (following PyMC documentation)
        data_vars = {}
        for i, name in enumerate(feature_names):
            var_name = name.lower().replace(" ", "_")
            # Handle PyMC version compatibility
            if hasattr(pm, 'ConstantData'):
                data_vars[var_name] = pm.ConstantData(var_name, X_array[:, i].astype(float))
            else:
                data_vars[var_name] = pm.Data(var_name, X_array[:, i].astype(float))
        
        # Response data
        if hasattr(pm, 'ConstantData'):
            RESPONSE_ID = pm.ConstantData("RESPONSE_ID", RESPONSE_ID)
        else:
            RESPONSE_ID = pm.Data("RESPONSE_ID", RESPONSE_ID)
        
        # Priors for coefficients (following PyMC documentation)
        beta_vars = {}
        for i, name in enumerate(feature_names):
            var_name = f"beta_{name.lower().replace(' ', '_')}"
            beta_vars[var_name] = pm.Normal(
                var_name, 
                priors["coef_mu"], 
                priors["coef_sigma"]
            )
        
        # FIXED: Cutpoints WITHOUT initvals - use informative priors instead
        # This is the key change that enables log-likelihood computation
        cutpoints = pm.Normal(
            "alpha",
            mu=np.arange(N_RESPONSE_CLASSES-1, dtype='float32') - (N_RESPONSE_CLASSES-2)/2,
            sigma=priors["cut_sigma"],
            transform=pm.distributions.transforms.ordered,
            shape=N_RESPONSE_CLASSES - 1,
            # NO INITVAL HERE! This enables log-likelihood computation
            dims="CUTPOINTS",
        )
        
        # Optional non-centered varying intercepts (hierarchical)
        if group_idx is not None and n_groups is not None:
            u_sigma = pm.HalfNormal("u_sigma", sigma=priors["u_sigma"])
            u_offset = pm.Normal("u_offset", mu=0.0, sigma=1.0, shape=n_groups)
            u = pm.Deterministic("u", u_offset * u_sigma)
            u_g = u[group_idx]
        else:
            u_g = 0
        
        # Linear predictor (following OC documentation)
        phi = sum(beta_vars[f"beta_{name.lower().replace(' ', '_')}"] * data_vars[name.lower().replace(" ", "_")] 
                for name in feature_names)
        
        # Add hierarchical intercept if specified
        if group_idx is not None and n_groups is not None:
            phi = phi + u_g
        
        # Likelihood (following OC documentation exactly) - maintain original variable name
        if link.lower() == "logit":
            pm.OrderedLogistic("response", cutpoints=cutpoints, eta=phi, observed=RESPONSE_ID)
        elif link.lower() == "probit":
            pm.OrderedProbit("response", cutpoints=cutpoints, eta=phi, observed=RESPONSE_ID)
        else:
            raise ValueError(f"Unsupported link function: {link}. Use 'logit' or 'probit'.")
    
    return model


def run_counterfactual_analysis(
    model: pm.Model,
    idata: Any,
    scenarios: Dict[str, Dict[str, int]],
    feature_names: list,
    n_samples: int = 1000
) -> Dict[str, Any]:
    """
    Run counterfactual analysis following PyMC Ordered Categories documentation exactly.
    
    This function implements the counterfactual analysis pattern shown
    in the PyMC Ordered Categories documentation, using pm.set_data() with
    xr.DataArray and pm.sample_posterior_predictive() to generate predictions.
    
    Parameters
    ----------
    model : pm.Model
        The fitted cumulative model with pm.Data() containers
    idata : az.InferenceData  
        Posterior samples from the model
    scenarios : Dict[str, Dict[str, int]]
        Dictionary mapping scenario names to feature values.
        Example: {"high_contact": {"contact": 1, "action": 1, "intention": 0}}
    feature_names : list
        Names of features in the model
    n_samples : int, default=1000
        Number of posterior predictive samples per scenario
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with counterfactual results including:
        - "scenarios": The input scenarios
        - "predictions": Posterior predictive samples for each scenario
        - "probabilities": Category probabilities for each scenario
        - "summary": Summary statistics
        
    Examples
    --------
    >>> scenarios = {
    ...     "baseline": {"action": 0, "intention": 0, "contact": 0},
    ...     "high_risk": {"action": 1, "intention": 1, "contact": 1}
    ... }
    >>> results = run_counterfactual_analysis(model, idata, scenarios, feature_names)
    """
    results = {
        "scenarios": scenarios,
        "predictions": {},
        "probabilities": {},
        "summary": {}
    }
    
    # Original data containers (to restore later)
    original_data = {}
    
    with model:
        # Store original data values
        for name in feature_names:
            var_name = name.lower().replace(" ", "_")
            if var_name in model.named_vars:
                original_data[var_name] = model[var_name].get_value()
        
        # Run counterfactuals for each scenario
        for scenario_name, scenario_values in scenarios.items():
            print(f"Running counterfactual for scenario: {scenario_name}")
            
            # Set data for this scenario (single observation following OC docs)
            scenario_data = {}
            for name in feature_names:
                var_name = name.lower().replace(" ", "_")
                value = scenario_values.get(name, 0)  # Default to 0 if not specified
                scenario_data[var_name] = np.array([value], dtype=float)
            
            # Update model data using pm.set_data (OC documentation pattern)
            pm.set_data(scenario_data, model=model)
            
            # Sample posterior predictive for this scenario
            ppc = pm.sample_posterior_predictive(
                idata,
                predictions=True,
                extend_inferencedata=False,
                random_seed=42
            )
            
            # Store predictions  
            pred_var = None
            for var in ppc.predictions.data_vars:
                if "response" in var:
                    pred_var = var
                    break
            
            if pred_var:
                predictions = ppc.predictions[pred_var].values.flatten()
                results["predictions"][scenario_name] = predictions
                
                # Calculate category probabilities
                unique_vals, counts = np.unique(predictions, return_counts=True)
                probs = counts / len(predictions)
                prob_dict = {int(val): float(prob) for val, prob in zip(unique_vals, probs)}
                results["probabilities"][scenario_name] = prob_dict
                
                # Summary statistics
                results["summary"][scenario_name] = {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "median": float(np.median(predictions)),
                    "mode": int(unique_vals[np.argmax(counts)])
                }
            else:
                print(f"Warning: Could not find prediction variable for scenario {scenario_name}")
        
        # Restore original data
        if original_data:
            pm.set_data(original_data, model=model)
    
    # For backward compatibility with PreMeet.ipynb, also return the old format
    # where each scenario directly contains mean, std, etc.
    legacy_results = {}
    for scenario_name, scenario_data in results["predictions"].items():
        if scenario_name in results["summary"]:
            legacy_results[scenario_name] = {
                "predictions": scenario_data,
                "mean": results["summary"][scenario_name]["mean"],
                "std": results["summary"][scenario_name]["std"],
                "percentiles": np.percentile(scenario_data, [2.5, 25, 50, 75, 97.5])
            }
    
    return legacy_results


def plot_counterfactual_results(
    results: Dict[str, Any],
    figsize: Tuple[float, float] = (12, 8)
) -> None:
    """
    Plot counterfactual analysis results following OC documentation exactly.
    
    This function creates histograms using plt.bar() exactly like the
    PyMC Ordered Categories documentation.
    
    Parameters
    ----------
    results : dict
        Results from run_counterfactual_analysis()
    figsize : tuple
        Figure size
        
    Examples
    --------
    >>> results = run_counterfactual_analysis(model, idata, scenarios, feature_names)
    >>> plot_counterfactual_results(results)
    """
    import matplotlib.pyplot as plt
    
    n_scenarios = len(results)
    
    # Handle different result formats
    if "predictions" in results:
        # New format from updated function
        scenarios_data = results.get("predictions", {})
    else:
        # Legacy format - results is directly the scenarios dict
        scenarios_data = results
    
    if not scenarios_data:
        print("No scenario data found to plot")
        return
    
    # Create subplots
    n_cols = min(4, len(scenarios_data))
    n_rows = (len(scenarios_data) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (scenario_name, scenario_data) in enumerate(scenarios_data.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Handle different data formats
            if isinstance(scenario_data, dict):
                if "predictions" in scenario_data:
                    predictions = scenario_data["predictions"]
                    mean_val = scenario_data.get("mean", np.mean(predictions))
                else:
                    predictions = scenario_data
                    mean_val = np.mean(predictions)
            else:
                predictions = scenario_data
                mean_val = np.mean(predictions)
            
            # Create histogram
            counts, bins = np.histogram(predictions, bins=np.arange(0, 8))  # 0-based categories
            total_counts = counts.sum()
            if total_counts > 0:
                counts = [c / total_counts for c in counts]
                ax.bar(bins[:-1], counts, width=0.8, alpha=0.7, 
                      color='skyblue', edgecolor='black')
            
            ax.set_xlim([-0.5, 6.5])
            ax.set_xlabel("Response Category")
            ax.set_ylabel("Density")
            ax.set_title(f"{scenario_name}\nMean: {mean_val:.2f}")
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(scenarios_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# Maintain compatibility with existing workflow functions
def get_model_structure(model: pm.Model) -> Dict[str, Any]:
    """
    Extract model structure information for diagnostics and visualization.
    
    This function provides compatibility with existing workflow functions
    by extracting key model information in a standardized format.
    """
    structure = {
        "free_vars": [rv.name for rv in model.free_RVs],
        "observed_vars": [rv.name for rv in model.observed_RVs],
        "coords": getattr(model, 'coords', {}),
        "name": getattr(model, 'name', 'unnamed_model')
    }
    
    return structure