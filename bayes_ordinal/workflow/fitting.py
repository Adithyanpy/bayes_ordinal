import pymc as pm
import arviz as az
from pymc.exceptions import SamplingError
from typing import Optional

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
    **sample_kwargs
) -> az.InferenceData:
    """
    Fit a PyMC model using advanced sampling methods.

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
    jitter_max_retries : int, default=5
        Number of retries if initialization fails.
    sampler : {"nuts", "smc", "approx"}, default="nuts"
        Sampling method to use:
        - "nuts": Standard NUTS/HMC (most reliable)
        - "smc": Sequential Monte Carlo (good for complex posteriors)
        - "approx": Approximate sampling (fast exploration)
    **sample_kwargs : dict
        Additional keyword arguments passed to the sampling function.

    Returns
    -------
    idata : az.InferenceData
        ArviZ InferenceData with posterior and sample statistics.

    Examples
    --------
    >>> idata = fit_model(my_model, draws=2000, tune=1000)
    >>> idata = fit_model(my_model, sampler="smc", draws=1000)
    >>> az.plot_trace(idata)
    """
    # Determine effective initvals (user or model default)
    default_iv = getattr(model, "_default_initvals", None)
    effective_initvals = initvals if initvals is not None else default_iv

    with model:
        if sampler == "nuts":
            # Standard NUTS sampling (PyMCOrdinal-compatible)
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
            
            # Only add log_likelihood if no initval conflicts
            try:
                # Check if model has initval parameters that would conflict
                has_initval = any(v is not None for v in model.rvs_to_initial_values.values())
                if not has_initval:
                    kwargs["idata_kwargs"] = {"log_likelihood": True}
            except:
                # If we can't check, don't add log_likelihood to be safe
                pass
            if effective_initvals is not None:
                kwargs["initvals"] = effective_initvals
            return pm.sample(**kwargs)
            
        elif sampler == "smc":
            # Sequential Monte Carlo for complex posteriors
            kwargs = dict(
                draws=draws,
                **sample_kwargs
            )
            idata = pm.sample_smc(**kwargs)
            if return_inferencedata:
                return idata
            else:
                return idata
                
        elif sampler == "approx":
            # Approximate sampling for quick exploration
            kwargs = dict(
                draws=draws,
                **sample_kwargs
            )
            idata = pm.sample_approx(**kwargs)
            if return_inferencedata:
                return idata
            else:
                return idata
            
        else:
            raise ValueError(f"Unknown sampler {sampler!r}; choose 'nuts', 'smc', or 'approx'")
