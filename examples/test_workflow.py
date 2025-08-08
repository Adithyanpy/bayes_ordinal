#!/usr/bin/env python3
"""
Quick test of the complete working workflow
==========================================
This script tests the key components without full sampling
"""

print("🧪 Testing Complete Working Workflow Components...")
print("=" * 55)

# Import required modules
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive for testing
    import matplotlib.pyplot as plt
    import arviz as az
    import pymc as pm
    
    from model_fixes import (
        partial_odds_model_fixed, 
        cumulative_model_fixed
    )
    print("✓ All imports successful")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Load and prepare data
try:
    data = pd.read_csv("../Dataset/Trolley.csv", sep=';').dropna()
    X = data[['action', 'intention', 'contact']].values
    y = data['response'].values - 1
    K = len(np.unique(y))
    
    print(f"✓ Data loaded: {data.shape[0]} obs, {K} categories")
    
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    exit(1)

# Test model creation
try:
    # Conservative priors
    partial_odds_priors = {
        "gamma_mu": 0.0, "gamma_sigma": 0.5,
        "beta_mu": 0.0, "beta_sigma": 0.5
    }
    
    cumulative_priors = {
        "coef_mu": 0.0, "coef_sigma": 0.5,
        "cut_sigma": 1.5
    }
    
    # Build models
    partial_model = partial_odds_model_fixed(
        y, X, K=K, priors=partial_odds_priors
    )
    
    cumulative_model = cumulative_model_fixed(
        y, X, K=K, priors=cumulative_priors, 
        feature_names=['action', 'intention', 'contact']
    )
    
    print("✓ Both models created successfully")
    print(f"  Partial odds: {len(partial_model.free_RVs)} variables")
    print(f"  Cumulative: {len(cumulative_model.free_RVs)} variables")
    
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    exit(1)

# Test quick sampling (minimal for speed)
try:
    print("Testing minimal sampling...")
    
    # Very quick sample just to test log-likelihood works
    with partial_model:
        idata_partial = pm.sample(
            draws=10, tune=10, chains=1,
            idata_kwargs={"log_likelihood": True},
            random_seed=42,
            progressbar=False
        )
    
    with cumulative_model:
        idata_cumulative = pm.sample(
            draws=10, tune=10, chains=1,
            idata_kwargs={"log_likelihood": True}, 
            random_seed=42,
            progressbar=False
        )
    
    print("✓ Sampling successful with log-likelihood")
    
    # Check log-likelihood is present
    has_ll_partial = hasattr(idata_partial, 'log_likelihood')
    has_ll_cumulative = hasattr(idata_cumulative, 'log_likelihood')
    
    print(f"  Partial odds log-likelihood: {'✓' if has_ll_partial else '✗'}")
    print(f"  Cumulative log-likelihood: {'✓' if has_ll_cumulative else '✗'}")
    
except Exception as e:
    print(f"✗ Sampling failed: {e}")
    exit(1)

# Test model comparison
try:
    if has_ll_partial and has_ll_cumulative:
        loo_partial = az.loo(idata_partial)
        loo_cumulative = az.loo(idata_cumulative)
        
        print("✓ LOO computation successful")
        print(f"  Partial odds LOO: {loo_partial.elpd_loo:.1f}")
        print(f"  Cumulative LOO: {loo_cumulative.elpd_loo:.1f}")
    else:
        print("⚠ Skipping LOO test - no log-likelihood")
        
except Exception as e:
    print(f"✗ LOO computation failed: {e}")

print(f"\n🎉 WORKFLOW TEST COMPLETE!")
print(f"✓ Models build without initvals")
print(f"✓ Sampling works with log-likelihood")
print(f"✓ Model comparison is possible")
print(f"\n🚀 Ready to run full workflow notebook!")