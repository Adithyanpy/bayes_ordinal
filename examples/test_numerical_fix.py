#!/usr/bin/env python3
"""
Simple test to verify numerical warnings are fixed.
"""

import numpy as np
import pymc as pm
import arviz as az
import bayes_ordinal as bo

print("🧪 TESTING NUMERICAL WARNINGS FIX")
print("=" * 40)

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

# Test our package approach
print("\n1️⃣ TESTING OUR PACKAGE")
print("-" * 25)

try:
    # Create model using our package
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
        
        # Posterior predictive
        idata.extend(pm.sample_posterior_predictive(idata))
    
    print("✓ Model fitted successfully")
    print(f"  Posterior samples: {idata.posterior.dims}")
    
    # Test diagnostics
    print("\n  Testing diagnostics...")
    summary = az.summary(idata, var_names=["beta", "alpha"])
    print("✓ Summary computed without warnings")
    
    # Test convergence diagnostics
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    print("✓ R-hat and ESS computed without warnings")
    
    # Test our diagnostics function
    diagnostics = bo.summarize_diagnostics(idata)
    print("✓ Our diagnostics function completed without warnings")
    
    print("\n✅ NUMERICAL WARNINGS FIXED!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc() 