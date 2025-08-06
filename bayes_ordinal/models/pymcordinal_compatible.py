"""
PyMCOrdinal-compatible model implementations.

This module provides exact replicas of PyMCOrdinal's model specifications
to ensure full compatibility and comparison capabilities.
"""

import pymc as pm
import pytensor.tensor as pt
import numpy as np
from typing import Dict, Any, Optional, Union

def constrained_uniform(N: int, min_val: float = 0, max_val: float = 1):
    """
    Exact replica of PyMCOrdinal's constrainedUniform function.
    
    Parameters
    ----------
    N : int
        Number of cutpoints (K-1 where K is number of categories).
    min_val : float, default=0
        Minimum value for cutpoints.
    max_val : float, default=1
        Maximum value for cutpoints.
    
    Returns
    -------
    cutpoints : pm.Deterministic
        Ordered cutpoints constrained to [min_val, max_val].
    """
    return pm.Deterministic(
        "cutpoints",
        pt.concatenate([
            np.ones(1) * min_val,
            pt.extra_ops.cumsum(pm.Dirichlet("cuts_unknown", a=np.ones(N - 2))) * (max_val - min_val) + min_val,
        ])
    )

def make_model(
    priors: Dict[str, Any],
    model_spec: int = 1,
    constrained_uniform: bool = False,
    logit: bool = True,
    df: Optional[Any] = None,
    y_col: str = "explicit_rating",
    X_cols: Optional[list] = None,
    K: Optional[int] = None
) -> pm.Model:
    """
    Exact replica of PyMCOrdinal's make_model function.
    
    Parameters
    ----------
    priors : dict
        Dictionary of prior hyperparameters.
    model_spec : int, default=1
        Model specification (1=salary only, 2=salary+worksat, 3=full model).
    constrained_uniform : bool, default=False
        Whether to use constrained Dirichlet cutpoints.
    logit : bool, default=True
        Whether to use logit (True) or probit (False) link.
    df : pandas.DataFrame, optional
        Dataframe containing the data.
    y_col : str, default="explicit_rating"
        Name of the response variable column.
    X_cols : list, optional
        List of feature column names.
    K : int, optional
        Number of categories.
    
    Returns
    -------
    model : pm.Model
        PyMC model object.
    """
    if df is None or X_cols is None or K is None:
        raise ValueError("df, X_cols, and K must be provided")
    
    with pm.Model() as model:
        # Cutpoints specification
        if constrained_uniform:
            cutpoints = constrained_uniform(K, 0, K)
        else:
            sigma = pm.Exponential("sigma", priors["sigma"])
            cutpoints = pm.Normal(
                "cutpoints",
                mu=priors["mu"],
                sigma=sigma,
                transform=pm.distributions.transforms.univariate_ordered,
            )

        # Model specification
        if model_spec == 1:
            beta = pm.Normal("beta", priors["beta"][0], priors["beta"][1], size=1)
            mu = pm.Deterministic("mu", beta[0] * df[X_cols[0]])
        elif model_spec == 2:
            beta = pm.Normal("beta", priors["beta"][0], priors["beta"][1], size=2)
            mu = pm.Deterministic("mu", beta[0] * df[X_cols[0]] + beta[1] * df[X_cols[1]])
        elif model_spec == 3:
            beta = pm.Normal("beta", priors["beta"][0], priors["beta"][1], size=3)
            mu = pm.Deterministic(
                "mu", beta[0] * df[X_cols[0]] + beta[1] * df[X_cols[1]] + beta[2] * df[X_cols[2]]
            )
        else:
            raise ValueError(f"Unknown model_spec {model_spec}")
        
        # Likelihood
        if logit:
            y_ = pm.OrderedLogistic("y", cutpoints=cutpoints, eta=mu, observed=df[y_col])
        else:
            y_ = pm.OrderedProbit("y", cutpoints=cutpoints, eta=mu, observed=df[y_col])
    
    return model

def sample_model(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 4,
    random_seed: int = 42,
    nuts_sampler: str = "numpyro",
    return_inferencedata: bool = True,
    log_likelihood: bool = True
) -> Any:
    """
    Sample from model using PyMCOrdinal's exact approach.
    
    Parameters
    ----------
    model : pm.Model
        PyMC model to sample from.
    draws : int, default=1000
        Number of draws.
    tune : int, default=500
        Number of tuning steps.
    chains : int, default=4
        Number of chains.
    random_seed : int, default=42
        Random seed.
    nuts_sampler : str, default="numpyro"
        NUTS sampler to use.
    return_inferencedata : bool, default=True
        Whether to return InferenceData.
    log_likelihood : bool, default=True
        Whether to include log likelihood in InferenceData.
    
    Returns
    -------
    idata : InferenceData
        Inference data from sampling.
    """
    with model:
        if nuts_sampler == "numpyro":
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                nuts_sampler="numpyro",
                return_inferencedata=return_inferencedata,
                idata_kwargs={"log_likelihood": log_likelihood}
            )
        else:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                return_inferencedata=return_inferencedata
            )
        
        # Add posterior predictive samples
        idata.extend(pm.sample_posterior_predictive(idata))
    
    return idata

def create_pymcordinal_workflow(
    df: Any,
    y_col: str = "explicit_rating",
    X_cols: list = None,
    K: int = 10
) -> Dict[str, Any]:
    """
    Create complete PyMCOrdinal workflow with all model specifications.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data.
    y_col : str, default="explicit_rating"
        Name of the response variable column.
    X_cols : list, optional
        List of feature column names.
    K : int, default=10
        Number of categories.
    
    Returns
    -------
    results : dict
        Dictionary containing all models and inference data.
    """
    if X_cols is None:
        X_cols = ["salary", "work_sat", "work_from_home"]
    
    # Define priors (exact PyMCOrdinal structure)
    priors = {
        "sigma": 1,
        "beta": [0, 1],
        "mu": np.linspace(0, K, K - 1)
    }
    
    results = {}
    
    # Model 1: Salary only (unconstrained)
    print("Creating Model 1: Salary only (unconstrained)")
    model1 = make_model(priors, model_spec=1, constrained_uniform=False, logit=True, 
                       df=df, y_col=y_col, X_cols=[X_cols[0]], K=K)
    idata1 = sample_model(model1)
    results["model1"] = {"model": model1, "idata": idata1}
    
    # Model 2: Salary + Work Satisfaction (unconstrained)
    print("Creating Model 2: Salary + Work Satisfaction (unconstrained)")
    model2 = make_model(priors, model_spec=2, constrained_uniform=False, logit=True,
                       df=df, y_col=y_col, X_cols=X_cols[:2], K=K)
    idata2 = sample_model(model2)
    results["model2"] = {"model": model2, "idata": idata2}
    
    # Model 3: Full model (unconstrained)
    print("Creating Model 3: Full model (unconstrained)")
    model3 = make_model(priors, model_spec=3, constrained_uniform=False, logit=True,
                       df=df, y_col=y_col, X_cols=X_cols, K=K)
    idata3 = sample_model(model3)
    results["model3"] = {"model": model3, "idata": idata3}
    
    # Model 4: Full model (constrained logit)
    print("Creating Model 4: Full model (constrained logit)")
    model4 = make_model(priors, model_spec=3, constrained_uniform=True, logit=True,
                       df=df, y_col=y_col, X_cols=X_cols, K=K)
    idata4 = sample_model(model4)
    results["model4"] = {"model": model4, "idata": idata4}
    
    # Model 5: Full model (constrained probit)
    print("Creating Model 5: Full model (constrained probit)")
    model5 = make_model(priors, model_spec=3, constrained_uniform=True, logit=False,
                       df=df, y_col=y_col, X_cols=X_cols, K=K)
    idata5 = sample_model(model5)
    results["model5"] = {"model": model5, "idata": idata5}
    
    return results 