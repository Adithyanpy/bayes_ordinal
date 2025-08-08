import pymc as pm
import arviz as az
from pymc.exceptions import SamplingError
from typing import Optional
import warnings

def fit_model(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    return_inferencedata: bool = True,
    init: str = "jitter+adapt_diag",
    initvals: dict | None = None,
    jitter_max_retries: int = 5,
    sampler: str = "nuts",
    nuts_sampler: str = "pymc",
    random_seed: int = 42,
    enable_log_likelihood: bool = True,
    **sample_kwargs
) -> az.InferenceData:
    """
    Fit a PyMC model using advanced sampling methods - OPTIMIZED FOR FIXED MODELS.

    This version is optimized to work with the fixed ordinal regression models
    that enable log-likelihood computation without initvals conflicts.

    Supports multiple sampling strategies:
    - "nuts": Standard NUTS/HMC (default, most reliable)
    - "smc": Sequential Monte Carlo for complex posteriors
    - "approx": Approximate sampling for quick exploration

    Parameters
    ----------
    model : pm.Model
        A fully specified PyMC model.
    draws : int, default=1000
        Number of posterior draws to sample.
    tune : int, default=1000
        Number of tuning steps (for NUTS).
    chains : int, default=4
        Number of MCMC chains to run.
    return_inferencedata : bool, default=True
        Whether to return ArviZ InferenceData.
    init : str, default="jitter+adapt_diag"
        Initialization method passed to `pm.sample`.
    initvals : dict, optional
        Dictionary of variable initial values.
        Note: Fixed models don't need this - conservative priors handle initialization.
    jitter_max_retries : int, default=5
        Number of retries if initialization fails.
    sampler : {"nuts", "smc", "approx"}, default="nuts"
        Sampling method to use:
        - "nuts": Standard NUTS/HMC (most reliable)
        - "smc": Sequential Monte Carlo (good for complex posteriors)
        - "approx": Approximate sampling (fast exploration)
    nuts_sampler : {"pymc", "numpyro"}, default="pymc"
        NUTS implementation to use. "numpyro" can handle complex models better
        but requires JAX installation.
    enable_log_likelihood : bool, default=True
        Whether to compute log-likelihood for model comparison.
        Fixed models support this without conflicts.
    **sample_kwargs : dict
        Additional keyword arguments passed to the sampling function.

    Returns
    -------
    idata : az.InferenceData
        ArviZ InferenceData with posterior and sample statistics.

    Notes
    -----
    This optimized version:
    - Automatically enables log-likelihood computation for fixed models
    - Handles initvals conflicts intelligently
    - Provides better error messages and fallback strategies
    - Optimized for the new fixed ordinal regression models

    Examples
    --------
    >>> # Standard usage with fixed models
    >>> idata = fit_model(my_model, draws=2000, tune=1000)
    >>> # Enable model comparison
    >>> idata = fit_model(my_model, enable_log_likelihood=True)
    >>> # Use NumPyro for complex models (requires JAX)
    >>> idata = fit_model(my_model, nuts_sampler="numpyro")
    """
    
    # Check if model uses fixed implementation (no default initvals)
    has_default_initvals = hasattr(model, '_default_initvals')
    uses_initvals_in_model = any(v is not None for v in model.rvs_to_initial_values.values())
    
    # Determine effective initvals (user or model default)
    default_iv = getattr(model, "_default_initvals", None)
    effective_initvals = initvals if initvals is not None else default_iv

    with model:
        if sampler == "nuts":
            # Standard NUTS sampling - optimized for fixed models
            kwargs = dict(
                draws=draws,
                tune=tune,
                chains=chains,
                init=init,
                jitter_max_retries=jitter_max_retries,
                return_inferencedata=return_inferencedata,
                nuts_sampler=nuts_sampler,
                random_seed=random_seed,
                **sample_kwargs
            )
            
            # Smart log-likelihood handling for fixed models
            if enable_log_likelihood:
                if not has_default_initvals and not uses_initvals_in_model:
                    # Fixed models - log-likelihood should work
                    kwargs["idata_kwargs"] = {"log_likelihood": True}
                    print("✓ Fixed model detected - enabling log-likelihood computation")
                elif nuts_sampler == "numpyro":
                    # NumPyro can handle models with initvals
                    kwargs["idata_kwargs"] = {"log_likelihood": True}
                    print("✓ NumPyro sampler - enabling log-likelihood computation")
                else:
                    # Legacy models with initvals - warn user
                    warnings.warn(
                        "Model appears to use initvals which may conflict with log-likelihood computation. "
                        "Consider using fixed model implementations or nuts_sampler='numpyro'.",
                        UserWarning
                    )
                    print("⚠ Initvals detected - skipping log-likelihood to avoid conflicts")
            
            # Add initvals if provided and needed
            if effective_initvals is not None and not has_default_initvals:
                kwargs["initvals"] = effective_initvals
                print("✓ Using provided initvals")
            elif has_default_initvals:
                print("⚠ Model has default initvals - may conflict with log-likelihood")
            
            try:
                return pm.sample(**kwargs)
            except Exception as e:
                if "log_likelihood" in str(e) and enable_log_likelihood:
                    print("⚠ Log-likelihood computation failed - retrying without it")
                    # Remove log_likelihood and try again
                    kwargs.pop("idata_kwargs", None)
                    return pm.sample(**kwargs)
                else:
                    raise e
            
        elif sampler == "smc":
            # Sequential Monte Carlo for complex posteriors
            kwargs = dict(
                draws=draws,
                **sample_kwargs
            )
            idata = pm.sample_smc(**kwargs)
            return idata
                
        elif sampler == "approx":
            # Approximate sampling for quick exploration
            kwargs = dict(
                draws=draws,
                **sample_kwargs
            )
            # Use ADVI as the default approximation method
            approx = pm.fit(method='advi', **kwargs)
            idata = approx.sample(draws=draws)
            return az.from_pymc3(idata) if hasattr(az, 'from_pymc3') else idata
            
        else:
            raise ValueError(f"Unknown sampler {sampler!r}; choose 'nuts', 'smc', or 'approx'")


def fit_model_with_comparison(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    **kwargs
) -> az.InferenceData:
    """
    Convenience function to fit a model with log-likelihood enabled for comparison.
    
    This is equivalent to fit_model(..., enable_log_likelihood=True) but makes
    the intent clear for model comparison workflows.
    
    Parameters
    ----------
    model : pm.Model
        PyMC model to fit
    draws, tune, chains : int
        Sampling parameters
    **kwargs
        Additional arguments passed to fit_model
        
    Returns
    -------
    az.InferenceData
        Fitted model with log-likelihood for comparison
        
    Examples
    --------
    >>> idata1 = fit_model_with_comparison(model1)
    >>> idata2 = fit_model_with_comparison(model2)
    >>> az.compare({"model1": idata1, "model2": idata2})
    """
    return fit_model(
        model,
        draws=draws,
        tune=tune,
        chains=chains,
        enable_log_likelihood=True,
        **kwargs
    )


def diagnose_fitting_issues(model: pm.Model) -> dict:
    """
    Diagnose potential fitting issues with a model.
    
    This function checks for common issues that can cause sampling problems
    and provides recommendations for fixing them.
    
    Parameters
    ----------
    model : pm.Model
        PyMC model to diagnose
        
    Returns
    -------
    dict
        Diagnosis results with recommendations
        
    Examples
    --------
    >>> diagnosis = diagnose_fitting_issues(my_model)
    >>> print(diagnosis['recommendations'])
    """
    issues = []
    recommendations = []
    
    # Check for initvals conflicts
    has_default_initvals = hasattr(model, '_default_initvals')
    uses_model_initvals = any(v is not None for v in model.rvs_to_initial_values.values())
    
    if has_default_initvals or uses_model_initvals:
        issues.append("Model uses initvals which may conflict with log-likelihood computation")
        recommendations.append("Consider using fixed model implementations for better compatibility")
    
    # Check model complexity
    n_free_vars = len(model.free_RVs)
    n_observed = len(model.observed_RVs)
    
    if n_free_vars > 50:
        issues.append(f"Model has many parameters ({n_free_vars})")
        recommendations.append("Consider using nuts_sampler='numpyro' for better performance")
    
    if n_observed == 0:
        issues.append("Model has no observed variables")
        recommendations.append("Ensure your model includes observed data")
    
    # Check for potential numerical issues
    for rv in model.free_RVs:
        if hasattr(rv, 'transform') and rv.transform is not None:
            if 'ordered' in str(rv.transform):
                if uses_model_initvals:
                    issues.append(f"Ordered variable {rv.name} uses initvals")
                    recommendations.append("Use informative priors instead of initvals for ordered variables")
    
    return {
        "has_issues": len(issues) > 0,
        "issues": issues,
        "recommendations": recommendations,
        "model_info": {
            "n_free_vars": n_free_vars,
            "n_observed": n_observed,
            "has_default_initvals": has_default_initvals,
            "uses_model_initvals": uses_model_initvals
        }
    }