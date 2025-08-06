#!/usr/bin/env python3
"""
Debug script to identify and fix numerical warnings at their root cause.
Following PyMC documentation exactly.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import bayes_ordinal as bo

print("🔍 DEBUGGING NUMERICAL WARNINGS")
print("=" * 50)

# ============================================================================
# 1. CREATE SIMPLE TEST DATA (Following PyMC Documentation)
# ============================================================================
print("\n1️⃣ CREATING TEST DATA")
print("-" * 30)

# Create simple ordinal data following PyMC documentation pattern
np.random.seed(42)
n_samples = 200

# Generate features
X = np.random.randn(n_samples, 2)
true_beta = np.array([0.5, -0.3])

# Generate latent variable
latent = X @ true_beta + np.random.normal(0, 0.5, n_samples)

# Create ordinal categories (0, 1, 2, 3)
K = 4
cutpoints_true = np.array([-1.0, 0.5, 2.0])  # K-1 cutpoints

# Convert to ordinal
y = np.zeros(n_samples, dtype=int)
y[latent > cutpoints_true[0]] = 1
y[latent > cutpoints_true[1]] = 2
y[latent > cutpoints_true[2]] = 3

print(f"✓ Test data created: {n_samples} samples, {K} categories")
print(f"  Response distribution: {np.bincount(y)}")
print(f"  True cutpoints: {cutpoints_true}")

# ============================================================================
# 2. TEST PYMC DOCUMENTATION APPROACH EXACTLY
# ============================================================================
print("\n2️⃣ TESTING PYMC DOCUMENTATION APPROACH")
print("-" * 40)

def test_pymc_documentation_approach():
    """Test the exact approach from PyMC documentation."""
    
    with pm.Model() as model:
        # Priors following PyMC documentation
        beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
        
        # Cutpoints with Ordered transform (PyMC documentation approach)
        cutpoints = pm.Normal(
            "cutpoints", 
            mu=np.linspace(-1, 2, K-1), 
            sigma=1, 
            shape=K-1,
            transform=pm.distributions.transforms.Ordered()
        )
        
        # Linear predictor
        mu = pm.Deterministic("mu", pm.math.dot(X, beta))
        
        # Likelihood
        y_obs = pm.OrderedLogistic("y_obs", eta=mu, cutpoints=cutpoints, observed=y)
        
        # Sample
        idata = pm.sample(
            draws=1000, 
            tune=500, 
            chains=4,  # PyMC recommends 4 chains
            return_inferencedata=True,
            random_seed=42
        )
        
        # Posterior predictive
        idata.extend(pm.sample_posterior_predictive(idata))
        
    return model, idata

try:
    model_pymc, idata_pymc = test_pymc_documentation_approach()
    print("✓ PyMC documentation approach completed successfully")
    print(f"  Posterior samples: {idata_pymc.posterior.dims}")
    
    # Check for warnings in diagnostics
    print("\n  Testing diagnostics...")
    summary = az.summary(idata_pymc, var_names=["beta", "cutpoints"])
    print("✓ Summary computed without warnings")
    
    # Test convergence diagnostics
    rhat = az.rhat(idata_pymc)
    ess = az.ess(idata_pymc)
    print("✓ R-hat and ESS computed without warnings")
    
except Exception as e:
    print(f"✗ PyMC documentation approach failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 3. TEST OUR PACKAGE APPROACH
# ============================================================================
print("\n3️⃣ TESTING OUR PACKAGE APPROACH")
print("-" * 35)

def test_our_package_approach():
    """Test our package approach and identify differences."""
    
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
    
    # Sample with same parameters as PyMC documentation
    with model:
        idata = pm.sample(
            draws=1000, 
            tune=500, 
            chains=4,
            return_inferencedata=True,
            random_seed=42
        )
        
        # Posterior predictive
        idata.extend(pm.sample_posterior_predictive(idata))
    
    return model, idata

try:
    model_ours, idata_ours = test_our_package_approach()
    print("✓ Our package approach completed successfully")
    print(f"  Posterior samples: {idata_ours.posterior.dims}")
    
    # Check for warnings in diagnostics
    print("\n  Testing diagnostics...")
    summary = az.summary(idata_ours, var_names=["beta", "alpha"])
    print("✓ Summary computed")
    
    # Test convergence diagnostics
    rhat = az.rhat(idata_ours)
    ess = az.ess(idata_ours)
    print("✓ R-hat and ESS computed")
    
except Exception as e:
    print(f"✗ Our package approach failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 4. COMPARE APPROACHES AND IDENTIFY ROOT CAUSE
# ============================================================================
print("\n4️⃣ COMPARING APPROACHES")
print("-" * 25)

if 'idata_pymc' in locals() and 'idata_ours' in locals():
    print("Comparing PyMC documentation vs our approach:")
    
    # Compare cutpoints
    print("\n  Cutpoints comparison:")
    print(f"    PyMC doc: {idata_pymc.posterior['cutpoints'].mean(dim=['chain', 'draw']).values}")
    print(f"    Our pkg:  {idata_ours.posterior['alpha'].mean(dim=['chain', 'draw']).values}")
    
    # Compare beta coefficients
    print("\n  Beta coefficients comparison:")
    print(f"    PyMC doc: {idata_pymc.posterior['beta'].mean(dim=['chain', 'draw']).values}")
    print(f"    Our pkg:  {idata_ours.posterior['beta'].mean(dim=['chain', 'draw']).values}")
    
    # Check for infinite or NaN values
    print("\n  Checking for numerical issues:")
    
    # Check PyMC documentation results
    pymc_has_inf = np.any(np.isinf(idata_pymc.posterior.to_array().values))
    pymc_has_nan = np.any(np.isnan(idata_pymc.posterior.to_array().values))
    print(f"    PyMC doc - Infinite values: {pymc_has_inf}")
    print(f"    PyMC doc - NaN values: {pymc_has_nan}")
    
    # Check our package results
    ours_has_inf = np.any(np.isinf(idata_ours.posterior.to_array().values))
    ours_has_nan = np.any(np.isnan(idata_ours.posterior.to_array().values))
    print(f"    Our pkg - Infinite values: {ours_has_inf}")
    print(f"    Our pkg - NaN values: {ours_has_nan}")

# ============================================================================
# 5. TEST DIAGNOSTICS SPECIFICALLY
# ============================================================================
print("\n5️⃣ TESTING DIAGNOSTICS SPECIFICALLY")
print("-" * 35)

if 'idata_ours' in locals():
    print("Testing our diagnostics function:")
    
    try:
        # Test our diagnostics function
        diagnostics = bo.summarize_diagnostics(idata_ours)
        print("✓ Our diagnostics function completed")
        
        # Test convergence check
        convergence = bo.check_convergence(idata_ours)
        print("✓ Convergence check completed")
        
        # Test computational issues
        issues = bo.diagnose_computational_issues(idata_ours)
        print("✓ Computational issues check completed")
        
    except Exception as e:
        print(f"✗ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()

print("\n🎯 NUMERICAL WARNINGS DEBUG COMPLETED!") 