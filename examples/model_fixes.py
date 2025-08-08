#!/usr/bin/env python3
"""
Model fixes to break the initvals vs log-likelihood loop
========================================================

Use these modified model functions in your existing notebooks
to enable proper model comparison with LOO/WAIC.
"""

import pymc as pm
import pytensor.tensor as pt
import numpy as np
from typing import Union, Optional, Dict, Any, List


def partial_odds_model_fixed(
    y: Union[np.ndarray, list[int]],
    X: np.ndarray,
    K: int | None = None,
    priors: dict | None = None,
    group_idx: np.ndarray | None = None,
    n_groups: int | None = None,
    model_name: str = "partial_odds_fixed",
) -> pm.Model:
    """
    Fixed partial-odds model WITHOUT default initvals.
    
    This version enables log-likelihood computation for model comparison
    while maintaining numerical stability through conservative priors.
    """
    
    # Infer K if not provided
    if K is None:
        K = len(np.unique(y))
    
    priors = priors or {}
    # Use more conservative defaults to prevent -inf without initvals
    mu_g = priors.get("gamma_mu", 0.0)
    sigma_g = priors.get("gamma_sigma", 0.5)  # More conservative than 5.0
    mu_b = priors.get("beta_mu", 0.0)
    sigma_b = priors.get("beta_sigma", 0.5)  # More conservative than 5.0
    u_sigma_pr = priors.get("u_sigma", 1.0)

    with pm.Model(name=model_name) as model:
        # 1) Adjacent-category intercepts with conservative priors
        gamma = pm.Normal("gamma", mu=mu_g, sigma=sigma_g, shape=K-1)

        # 2) Fixed slopes with conservative priors
        beta = pm.Normal("beta", mu=mu_b, sigma=sigma_b, shape=X.shape[1])

        # 3) Optional non-centered varying intercepts
        if group_idx is not None and n_groups is not None:
            u_sigma = pm.HalfNormal("u_sigma", sigma=u_sigma_pr)
            u_offset = pm.Normal("u_offset", mu=0.0, sigma=1.0, shape=n_groups)
            u = pm.Deterministic("u", u_offset * u_sigma)
            u_g = u[group_idx][:, None]
        else:
            u_g = 0

        # 4) Build η matrix (n × (K−1))
        η = X @ beta
        η = η[:, None] + gamma
        η = η + u_g

        # 5) Stick-breaking to get category probabilities
        θ = pm.math.sigmoid(η)
        rem = pt.ones_like(θ[:, :1])
        ps = []
        for j in range(K-1):
            pj = rem * θ[:, j : j+1]
            ps.append(pj)
            rem = rem * (1 - θ[:, j : j+1])
        ps.append(rem)
        p = pt.concatenate(ps, axis=1)

        # 6) Likelihood
        pm.Categorical("y_obs", p=p, observed=y)

        # ── NO DEFAULT INITVALS! This enables log-likelihood computation ──
        # Conservative priors should prevent -inf values instead

    return model


def cumulative_model_fixed(
    y: Union[np.ndarray, list],
    X: Union[np.ndarray, list],
    K: Optional[int] = None,
    link: str = "logit",
    priors: Optional[Dict[str, Any]] = None,
    model_name: str = "cumulative_fixed",
    feature_names: Optional[List[str]] = None,
    group_idx: Optional[np.ndarray] = None,
    n_groups: Optional[int] = None,
) -> pm.Model:
    """
    Fixed cumulative model WITHOUT initvals on cutpoints.
    
    This version enables log-likelihood computation for model comparison
    while maintaining numerical stability through informed priors.
    """
    
    # Convert inputs to numpy arrays
    y_array = np.array(y)
    X_array = np.array(X)
    
    if K is None:
        K = len(np.unique(y_array))
    
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_array.shape[1])]
    
    # Default priors
    if priors is None:
        priors = {
            "coef_mu": 0.0,
            "coef_sigma": 0.5,  # More conservative
            "cut_mu": 0.0,
            "cut_sigma": 1.5,   # Slightly wider for stability
            "u_sigma": 1.0
        }

    with pm.Model(name=model_name) as model:
        # Create data containers (compatible with older PyMC versions)
        data_vars = {}
        for i, name in enumerate(feature_names):
            clean_name = name.lower().replace(" ", "_")
            if hasattr(pm, 'ConstantData'):
                data_vars[clean_name] = pm.ConstantData(clean_name, X_array[:, i])
            else:
                data_vars[clean_name] = pm.Data(clean_name, X_array[:, i])
        
        if hasattr(pm, 'ConstantData'):
            RESPONSE_ID = pm.ConstantData("RESPONSE_ID", y_array)
        else:
            RESPONSE_ID = pm.Data("RESPONSE_ID", y_array)
        N_RESPONSE_CLASSES = K
        
        # Add cutpoints dimension
        model.add_coord("CUTPOINTS", values=list(range(N_RESPONSE_CLASSES - 1)))
        
        # Coefficients
        beta_vars = {}
        for i, name in enumerate(feature_names):
            var_name = f"beta_{name.lower().replace(' ', '_')}"
            beta_vars[var_name] = pm.Normal(
                var_name, 
                priors["coef_mu"], 
                priors["coef_sigma"]
            )
        
        # Cutpoints WITHOUT initvals - use informative prior instead
        # This is the key change that enables log-likelihood computation
        cutpoints = pm.Normal(
            "alpha",
            mu=np.arange(N_RESPONSE_CLASSES-1, dtype='float32') - (N_RESPONSE_CLASSES-2)/2,
            sigma=priors["cut_sigma"],
            transform=pm.distributions.transforms.ordered,
            shape=N_RESPONSE_CLASSES - 1,
            # NO INITVAL HERE!
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
        
        # Linear predictor
        phi = sum(beta_vars[f"beta_{name.lower().replace(' ', '_')}"] * data_vars[name.lower().replace(" ", "_")] 
                for name in feature_names)
        
        # Add hierarchical intercept if specified
        if group_idx is not None and n_groups is not None:
            phi = phi + u_g
        
        # Likelihood
        if link.lower() == "logit":
            pm.OrderedLogistic("y_obs", cutpoints=cutpoints, eta=phi, observed=RESPONSE_ID)
        elif link.lower() == "probit":
            pm.OrderedProbit("y_obs", cutpoints=cutpoints, eta=phi, observed=RESPONSE_ID)
        else:
            raise ValueError(f"Unsupported link function: {link}. Use 'logit' or 'probit'.")
    
    return model


def sample_with_log_likelihood(model, **sample_kwargs):
    """
    Sample a model with log-likelihood computation enabled.
    
    This function handles the sampling and log-likelihood computation
    in a way that should work with the fixed models.
    """
    
    # Set default sample parameters for stability
    default_kwargs = {
        'draws': 1000,
        'tune': 1000, 
        'chains': 2,
        'init': 'jitter+adapt_diag',
        'idata_kwargs': {"log_likelihood": True},
        'random_seed': 42
    }
    default_kwargs.update(sample_kwargs)
    
    print(f"Sampling {model.name} with log-likelihood computation...")
    
    with model:
        idata = pm.sample(**default_kwargs)
    
    # Verify log-likelihood was computed
    if hasattr(idata, 'log_likelihood'):
        ll_vars = list(idata.log_likelihood.data_vars.keys())
        print(f"✓ Log-likelihood computed for: {ll_vars}")
    else:
        print("⚠ Log-likelihood not found in InferenceData")
        
    return idata


def compare_models_with_loo_waic(idata1, idata2, model1_name="Model 1", model2_name="Model 2"):
    """
    Compare two models using both LOO and WAIC.
    
    This function tries both methods and reports which model is better.
    """
    
    print(f"\nComparing {model1_name} vs {model2_name}")
    print("=" * 50)
    
    # Try LOO comparison
    try:
        loo1 = az.loo(idata1)
        loo2 = az.loo(idata2)
        
        print(f"LOO Comparison:")
        print(f"  {model1_name}: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")
        print(f"  {model2_name}: {loo2.elpd_loo:.2f} ± {loo2.se:.2f}")
        
        if loo1.elpd_loo > loo2.elpd_loo:
            winner = model1_name
            diff = loo1.elpd_loo - loo2.elpd_loo
        else:
            winner = model2_name
            diff = loo2.elpd_loo - loo1.elpd_loo
            
        print(f"  ✓ {winner} shows better fit (Δ LOO = {diff:.2f})")
        
    except Exception as e:
        print(f"LOO comparison failed: {e}")
    
    # Try WAIC comparison
    try:
        waic1 = az.waic(idata1)
        waic2 = az.waic(idata2)
        
        print(f"\nWAIC Comparison:")
        print(f"  {model1_name}: {waic1.elpd_waic:.2f} ± {waic1.se:.2f}")
        print(f"  {model2_name}: {waic2.elpd_waic:.2f} ± {waic2.se:.2f}")
        
        if waic1.elpd_waic > waic2.elpd_waic:
            winner = model1_name
            diff = waic1.elpd_waic - waic2.elpd_waic
        else:
            winner = model2_name
            diff = waic2.elpd_waic - waic1.elpd_waic
            
        print(f"  ✓ {winner} shows better fit (Δ WAIC = {diff:.2f})")
        
    except Exception as e:
        print(f"WAIC comparison failed: {e}")


if __name__ == "__main__":
    print("Model fixes loaded successfully!")
    print("\nUsage:")
    print("1. Import these functions: from model_fixes import partial_odds_model_fixed, cumulative_model_fixed")
    print("2. Use them as drop-in replacements for your existing models")
    print("3. Enable log-likelihood: sample_with_log_likelihood(model)")
    print("4. Compare models: compare_models_with_loo_waic(idata1, idata2)")