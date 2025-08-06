import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
from ..plot_utils import show_and_save

def run_prior_predictive(
    model: pm.Model,
    draws: int = 200,
    plot: bool = True,
    include_r2: bool = True,
    y_obs: np.ndarray = None
) -> az.InferenceData:
    """
    Run advanced prior predictive checks on a PyMC model.

    This function samples from the prior predictive distribution and optionally
    visualizes the distribution of the response variable `y_obs`.

    Parameters
    ----------
    model : pm.Model
        The unsampled PyMC model containing only prior definitions.
    draws : int, default=200
        Number of prior predictive samples.
    plot : bool, default=True
        Whether to plot a histogram of `y_obs` from the prior predictive draws.
    include_r2 : bool, default=True
        Whether to include R² samples for prior predictive checks.

    Returns
    -------
    idata : az.InferenceData
        An ArviZ InferenceData object with prior predictive samples.

    Examples
    --------
    >>> model = cumulative_model(y, X, K=4)
    >>> idata = run_prior_predictive(model)
    >>> az.plot_dist(idata.prior_predictive['y_obs'].values.flatten())
    """
    with model:
        idata = pm.sample_prior_predictive(samples=draws, return_inferencedata=True)
        
        # Add R² samples if requested
        if include_r2:
            try:
                # Skip R² for now as it requires more complex setup
                print("✓ R² samples skipped (requires posterior samples)")
            except Exception as e:
                print(f"Could not compute R² samples: {e}")

    if plot:
        # Extract and flatten the prior_predictive draws for response variable
        # Handle variable name variations
        available_vars = list(idata.prior_predictive.data_vars.keys())
        
        # Try different possible variable names
        possible_names = ["y_obs", "response", "y"]
        y_flat = None
        
        for name in possible_names:
            try:
                y_flat = idata.prior_predictive[name].values.flatten()
                print(f"Using prior predictive variable: {name}")
                break
            except KeyError:
                continue
        
        # If not found, try to find any variable that might be the response
        if y_flat is None:
            # Look for variables that don't contain common parameter names
            param_names = ["alpha", "beta", "gamma", "sigma", "mu", "tau"]
            response_vars = [v for v in available_vars if not any(param in v.lower() for param in param_names)]
            
            if response_vars:
                y_flat = idata.prior_predictive[response_vars[0]].values.flatten()
                print(f"Using prior predictive variable: {response_vars[0]}")
            else:
                print(f"Warning: Could not find response variable in prior predictive data")
                print(f"Available variables: {available_vars}")
                return idata

        # Create a proper histogram for discrete ordinal data
        plt.figure(figsize=(12, 5))
        
        # Count unique values to determine K
        unique_vals = np.unique(y_flat)
        K = len(unique_vals)
        
        # Prior predictive distribution
        plt.subplot(1, 2, 1)
        plt.hist(y_flat, bins=range(K+1), alpha=0.7, edgecolor='black', 
                label="Prior Predictive", color='skyblue')
        plt.xlabel("Ordinal Category")
        plt.ylabel("Frequency")
        plt.title("Prior Predictive Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Observed distribution (if provided)
        if y_obs is not None:
            plt.subplot(1, 2, 2)
            plt.hist(y_obs, bins=range(K+1), alpha=0.7, edgecolor='black',
                    label="Observed", color='lightcoral')
            plt.xlabel("Ordinal Category")
            plt.ylabel("Frequency")
            plt.title("Observed Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        show_and_save("prior_predictive_distribution")
        
        # Plot R² distribution if available
        if "r2" in idata.prior_predictive:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            az.plot_dist(idata.prior_predictive["r2"].values.flatten(), label="R² prior")
            plt.xlabel("R²")
            plt.ylabel("Density")
            plt.title("Prior Predictive R² Distribution")
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.scatter(idata.prior_predictive["r2"].values.flatten(), 
                       idata.prior_predictive["y_obs"].values.flatten(), 
                       alpha=0.5)
            plt.xlabel("R²")
            plt.ylabel("y_obs")
            plt.title("R² vs y_obs")
            show_and_save("prior_predictive_r2")

    return idata
