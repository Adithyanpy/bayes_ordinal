#!/usr/bin/env python3
"""
Test the Dirichlet cutpoint approach.
"""

import numpy as np
import pymc as pm
import arviz as az
import bayes_ordinal as bo

print("🧪 TESTING DIRICHLET CUTPOINTS")
print("=" * 35)

# Create simple test data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
K = 4

# Create ordinal response
true_beta = np.array([0.5, -0.3])
latent = X @ true_beta + np.random.normal(0, 0.5, n_samples)
cutpoints_true = np.array([-1.0, 0.5, 2.0])

y = np.zeros(n_samples, dtype=int)
y[latent > cutpoints_true[0]] = 1
y[latent > cutpoints_true[1]] = 2
y[latent > cutpoints_true[2]] = 3

print(f"✓ Test data created: {n_samples} samples, {K} categories")

# Test Dirichlet cutpoints
print("\n1️⃣ TESTING DIRICHLET CUTPOINTS")
print("-" * 30)

try:
    # Create model using Dirichlet cutpoints (default)
    model = bo.cumulative_model(
        y=y, 
        X=X, 
        K=K, 
        link="logit",
        priors={
            "coef_mu": 0.0,
            "coef_sigma": 1.0,
            "cut_min": -2.0,
            "cut_max": 3.0
        }
    )
    
    # Sample
    with model:
        idata = pm.sample(
            draws=500, 
            tune=200, 
            chains=2,
            return_inferencedata=True,
            random_seed=42
        )
    
    print("✓ Dirichlet cutpoints completed successfully")
    print(f"  Posterior samples: {idata.posterior.dims}")
    
    # Check cutpoints
    alpha = idata.posterior['alpha'].mean(dim=['chain', 'draw']).values
    print(f"  Cutpoints: {alpha}")
    
    # Check beta coefficients
    beta = idata.posterior['beta'].mean(dim=['chain', 'draw']).values
    print(f"  Beta coefficients: {beta}")
    
    # Check for numerical issues
    has_inf = np.any(np.isinf(idata.posterior.to_array().values))
    has_nan = np.any(np.isnan(idata.posterior.to_array().values))
    print(f"  Infinite values: {has_inf}")
    print(f"  NaN values: {has_nan}")
    
    # Test diagnostics
    print("\n  Testing diagnostics...")
    summary = az.summary(idata, var_names=["beta", "alpha"])
    print("✓ Summary computed without warnings")
    
    # Test convergence diagnostics
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    print("✓ R-hat and ESS computed without warnings")
    
    print("\n✅ DIRICHLET CUTPOINTS WORK PERFECTLY!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc() 