import pymc as pm
import arviz as az
import numpy as np
from typing import Optional, Dict, Any

def fit_ordinal_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    return_inferencedata: bool = True,
    init: str = "jitter+adapt_diag",
    jitter_max_retries: int = 10,
    sampler: str = "nuts",
    nuts_sampler: str = "pymc",
    random_seed: Optional[int] = 42,
    enable_log_likelihood: bool = True,
    target_accept: float = 0.8,
    max_treedepth: int = 15,
    progressbar: bool = True,
    compute_convergence_checks: bool = True,
    **sample_kwargs
) -> az.InferenceData:
    """
    Fit a Bayesian ordinal regression model using NUTS sampling.
    
    This function is specifically designed for ordinal regression models (cumulative)
    and implements best practices for reliable inference:
    - Uses NUTS sampler with ordinal-optimized tuning parameters
    - Enables log-likelihood computation for LOO/WAIC model comparison
    - Implements robust initialization with multiple retry attempts
    - Sets conservative sampling parameters for ordinal models with cutpoints
    
    Parameters
    ----------
    model : pm.Model
        A PyMC ordinal regression model (e.g., from cumulative_model).
    draws : int, default=2000
        Number of posterior draws. Higher values recommended for ordinal models.
    tune : int, default=1000
        Number of tuning steps for NUTS adaptation.
    chains : int, default=4
        Number of MCMC chains for reliable inference.
    return_inferencedata : bool, default=True
        Whether to return ArviZ InferenceData.
    init : str, default="jitter+adapt_diag"
        Initialization method. Robust for ordinal models with cutpoints.
    jitter_max_retries : int, default=10
        Number of retries if initialization fails.
    sampler : str, default="nuts"
        Sampling method (NUTS is optimal for ordinal models).
    nuts_sampler : str, default="pymc"
        NUTS implementation to use.
    random_seed : Optional[int], default=42
        Random seed for reproducibility.
    enable_log_likelihood : bool, default=True
        Whether to compute log-likelihood for model comparison.
    target_accept : float, default=0.8
        Target acceptance rate for NUTS. Optimal for ordinal models.
    max_treedepth : int, default=15
        Maximum tree depth for NUTS. Higher for models with many cutpoints.
    progressbar : bool, default=True
        Whether to show sampling progress bar and chain information.
    compute_convergence_checks : bool, default=True
        Whether to compute and display convergence diagnostics during sampling.
    **sample_kwargs : dict
        Additional keyword arguments passed to pm.sample().

    Returns
    -------
    idata : az.InferenceData
        ArviZ InferenceData with posterior samples and log-likelihood.
        
    Notes
    -----
    Ordinal regression models require careful sampling due to cutpoint constraints:
    - draws >= 2000 for reliable posterior estimates of cutpoints
    - tune >= 1000 for proper NUTS adaptation to cutpoint geometry
    - target_accept = 0.8 for optimal mixing in constrained parameter space
    - max_treedepth = 15 for complex models with many ordinal categories
    """
    
    if sampler != "nuts":
        raise ValueError("Only NUTS sampler is supported for ordinal regression models")
    
    # Validate ordinal model structure
    if not _is_ordinal_model(model):
        raise ValueError("Model must be an ordinal regression model with 'y' as observed variable and cutpoints")
    
    with model:
        # NUTS sampling with ordinal-optimized parameters
        kwargs = dict(
            draws=draws,
            tune=tune,
            chains=chains,
            init=init,
            jitter_max_retries=jitter_max_retries,
            return_inferencedata=return_inferencedata,
            nuts_sampler=nuts_sampler,
            random_seed=random_seed,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            **sample_kwargs
        )
        
        # Enable log-likelihood for model comparison (essential for ordinal models)
        if enable_log_likelihood:
            kwargs["idata_kwargs"] = {"log_likelihood": True}
        
        try:
            # Ensure progress tracking and diagnostics are enabled
            if "progressbar" not in kwargs:
                kwargs["progressbar"] = True
            if "compute_convergence_checks" not in kwargs:
                kwargs["compute_convergence_checks"] = True
                
            return pm.sample(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Sampling failed for ordinal regression model: {e}. "
                             f"Common issues: insufficient tuning, poor initialization, or model specification. "
                             f"Try increasing tune, adjusting priors, or checking cutpoint constraints.")


def _is_ordinal_model(model: pm.Model) -> bool:
    """
    Check if the model is a valid Bayesian ordinal regression model.
    
    Parameters
    ----------
    model : pm.Model
        PyMC model to check
        
    Returns
    -------
    bool
        True if model has ordinal regression structure
    """
    # Check for observed ordinal variable (must be named 'y')
    observed_vars = [rv.name for rv in model.observed_RVs]
    if not any('y' in name for name in observed_vars):
        return False
    
    # Check for cutpoints (essential for ordinal regression)
    free_vars = [rv.name for rv in model.free_RVs]
    if not any('cutpoint' in name for name in free_vars):
        return False
    
    # Check for coefficients (beta parameters)
    if not any('beta' in name for name in free_vars):
        return False
    
    return True


