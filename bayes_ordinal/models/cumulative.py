"""
Cumulative (Proportional Odds) Ordinal Regression Model.

This module implements the cumulative ordinal regression model following
PyMC Ordered Categories (OC) documentation and McElreath's Statistical 
Rethinking principles. It provides exact implementations matching the
OC documentation structure and parameterization.

Key Features:
- Follows PyMC Ordered Categories documentation exactly
- Uses pm.Data() for flexible predictions
- Uses "response" variable name (not "y")
- Uses phi as linear predictor name
- Uses univariate_ordered transform with initval for cutpoints (exact OC documentation)
- Uses pm.OrderedLogistic() and pm.OrderedProbit() as in OC documentation
- Supports both logit and probit link functions
- Includes counterfactual analysis following OC documentation

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
    feature_names: Optional[list] = None
) -> pm.Model:
    """
    Build cumulative ordinal regression model following PyMC Ordered Categories documentation.
    
    This function creates a model that matches the PyMC Ordered Categories (OC) documentation
    structure exactly, using pm.Data() for inputs and pm.OrderedLogistic()/pm.OrderedProbit()
    for the likelihood. It follows the exact parameterization from the OC documentation.
    
    Key Features:
    - Uses pm.Data() for flexible predictions following OC documentation
    - Uses "response" as variable name (not "y") following OC documentation  
    - Uses phi as linear predictor name following OC documentation
    - Uses univariate_ordered transform with initval for cutpoints (exact OC documentation)
    - Follows exact parameterization from PyMC Ordered Categories documentation
    - Uses pm.OrderedLogistic() and pm.OrderedProbit() as in OC documentation
    
    Parameters
    ----------
    y : np.ndarray or pd.Series
        Response variable (will be factorized to 0-based indexing)
    X : np.ndarray or pd.DataFrame
        Feature matrix or DataFrame. If DataFrame, feature_names will be extracted.
    K : int, optional
        Number of response categories. If None, will be inferred from y.
    link : str, default="logit"
        Link function ("logit" or "probit")
    priors : dict, optional
        Prior specifications
    model_name : str, default="cumulative_model"
        Name for the PyMC model
    feature_names : list, optional
        Names of features for better variable naming. If None and X is DataFrame,
        column names will be used.
        
    Returns
    -------
    pm.Model
        PyMC model following OC documentation structure
        
    References
    ----------
    PyMC Ordered Categories Documentation: 
    https://docs.pymc.io/en/latest/learn/core_notebooks/ordinal_regression.html
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
    
    # Use default priors if none provided
    if priors is None:
        priors = {
            "coef_mu": 0.0,
            "coef_sigma": 0.5,  # Following PyMC documentation
            "cut_mu": 0.0,
            "cut_sigma": 1.0
        }
    
    with pm.Model(coords=coords, name=model_name) as model:
        
        # Data for posterior predictions (following PyMC documentation)
        data_vars = {}
        for i, name in enumerate(feature_names):
            var_name = name.lower().replace(" ", "_")
            data_vars[var_name] = pm.Data(var_name, X_array[:, i].astype(float))
        
        # Priors for coefficients (following PyMC documentation)
        beta_vars = {}
        for i, name in enumerate(feature_names):
            var_name = f"beta_{name.lower().replace(' ', '_')}"
            beta_vars[var_name] = pm.Normal(
                var_name, 
                priors["coef_mu"], 
                priors["coef_sigma"]
            )
        
        # Cutpoints following OC documentation exactly
        cutpoints = pm.Normal(
            "alpha",
            mu=0,
            sigma=1,
            transform=pm.distributions.transforms.univariate_ordered,
            shape=N_RESPONSE_CLASSES - 1,
            initval=np.arange(N_RESPONSE_CLASSES - 1) - 2.5,  # use ordering (with coarse log-odds centering) for init
            dims="CUTPOINTS",
        )
        
        # Linear predictor (following OC documentation)
        phi = sum(beta_vars[f"beta_{name.lower().replace(' ', '_')}"] * data_vars[name.lower().replace(" ", "_")] 
                for name in feature_names)
        
        # Likelihood (following OC documentation exactly)
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
    
    Key Features:
    - Uses pm.set_data() with xr.DataArray for counterfactual scenarios (exact OC documentation)
    - Uses "response" variable name and converts back to 1-7 scale (exact OC documentation)
    - Creates manual histograms with plt.bar() (exact OC documentation)
    - Provides comprehensive summary statistics
    
    Parameters
    ----------
    model : pm.Model
        Fitted PyMC model
    idata : Any
        Inference data from model fitting
    scenarios : dict
        Dictionary mapping scenario names to feature dictionaries (like OC documentation)
    feature_names : list
        Names of features (must match model data variables)
    n_samples : int, default=1000
        Number of posterior samples to use (not used in current implementation)
        
    Returns
    -------
    dict
        Counterfactual analysis results with predictions, mean, std, and percentiles
        
    References
    ----------
    PyMC Ordered Categories Documentation:
    https://docs.pymc.io/en/latest/learn/core_notebooks/ordinal_regression.html
    """
    
    results = {}
    # Get number of samples from inference data
    N_RESPONSES = idata.posterior.dims['chain'] * idata.posterior.dims['draw']
    
    for scenario_name, feature_values in scenarios.items():
        with model:
            # Set data using xr.DataArray like OC documentation exactly
            data_dict = {}
            for name in feature_names:
                var_name = name.lower().replace(" ", "_")
                # For counterfactual analysis, we only need one observation
                data_dict[var_name] = xr.DataArray([feature_values[var_name]])
            
            pm.set_data(data_dict)
            
            # Sample posterior predictive like OC documentation
            ppd = pm.sample_posterior_predictive(
                idata, 
                extend_inferencedata=False
            )
        
        # Get response variable name (with model prefix)
        response_var_name = None
        for var_name in ppd.posterior_predictive.data_vars:
            if "response" in var_name:
                response_var_name = var_name
                break
        
        if response_var_name is None:
            raise ValueError("Could not find response variable in posterior predictive data")
        
        # Recode back to 1-7 like OC documentation exactly
        posterior_predictive = ppd.posterior_predictive[response_var_name] + 1
        
        # Calculate summary statistics
        mean_val = posterior_predictive.values.mean()
        std_val = posterior_predictive.values.std()
        percentiles = np.percentile(posterior_predictive.values.flatten(), [2.5, 25, 50, 75, 97.5])
        
        results[scenario_name] = {
            "predictions": posterior_predictive.values.flatten(),
            "mean": mean_val,
            "std": std_val,
            "percentiles": percentiles
        }
    
    return results


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
    """
    import matplotlib.pyplot as plt
    
    n_scenarios = len(results)
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    
    for i, (scenario_name, result) in enumerate(results.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Create histogram like OC documentation exactly
            counts, bins = np.histogram(result['predictions'], bins=np.arange(1, 9))
            total_counts = counts.sum()
            counts = [c / total_counts for c in counts]
            ax.bar(bins[:-1], counts, width=0.25)
            
            ax.set_xlim([0.5, 7.5])
            ax.set_xlabel("response")
            ax.set_ylabel("density")
            ax.set_title(f"{scenario_name}\nMean: {result['mean']:.2f}")
    
    plt.tight_layout()
    plt.show()
    
    for i, (scenario_name, result) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot histogram of predictions
        ax.hist(result["predictions"], bins=range(1, 9), alpha=0.7, 
                density=True, color='skyblue', edgecolor='black')
        
        ax.set_xlabel("Response Category")
        ax.set_ylabel("Density")
        ax.set_title(f"Scenario: {scenario_name}")
        ax.set_xlim([0.5, 7.5])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
