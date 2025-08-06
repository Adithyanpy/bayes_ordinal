#!/usr/bin/env python3
"""
Simple test script to verify logit and probit functionality.
"""

import numpy as np
import pymc as pm
import arviz as az
import bayes_ordinal as bo

print("🧪 TESTING LOGIT AND PROBIT FUNCTIONALITY")
print("=" * 50)

# Create simple test data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)  # 2 features
K = 4  # 4 categories (0, 1, 2, 3)

# Create ordinal response
true_beta = np.array([0.5, -0.3])
eta = X @ true_beta
# Simple threshold model
y = np.zeros(n_samples, dtype=int)
y[eta > 0.5] = 1
y[eta > 1.0] = 2
y[eta > 1.5] = 3

print(f"✓ Test data created: {n_samples} samples, {K} categories")
print(f"  Response distribution: {np.bincount(y)}")

# Test 1: Logit model
print("\n1️⃣ TESTING LOGIT MODEL")
print("-" * 30)

try:
    # Create logit model
    model_logit = bo.cumulative_model(
        y=y, 
        X=X, 
        K=K, 
        link="logit",
        priors={"coef_sigma": 1.0, "cut_min": 0.0, "cut_max": 4.0}
    )
    print("✓ Logit model created successfully")
    
    # Fit with minimal samples for testing
    with model_logit:
        idata_logit = pm.sample(
            draws=100, 
            chains=2, 
            tune=50,
            return_inferencedata=True,
            random_seed=42
        )
    print("✓ Logit model fitted successfully")
    print(f"  Posterior samples: {idata_logit.posterior.dims}")
    
except Exception as e:
    print(f"✗ Logit model failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Probit model
print("\n2️⃣ TESTING PROBIT MODEL")
print("-" * 30)

try:
    # Create probit model
    model_probit = bo.cumulative_model(
        y=y, 
        X=X, 
        K=K, 
        link="probit",
        priors={"coef_sigma": 1.0, "cut_min": 0.0, "cut_max": 4.0}
    )
    print("✓ Probit model created successfully")
    
    # Fit with minimal samples for testing
    with model_probit:
        idata_probit = pm.sample(
            draws=100, 
            chains=2, 
            tune=50,
            return_inferencedata=True,
            random_seed=42
        )
    print("✓ Probit model fitted successfully")
    print(f"  Posterior samples: {idata_probit.posterior.dims}")
    
except Exception as e:
    print(f"✗ Probit model failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Basic diagnostics
print("\n3️⃣ TESTING BASIC DIAGNOSTICS")
print("-" * 30)

try:
    if 'idata_logit' in locals():
        summary_logit = az.summary(idata_logit, var_names=["beta", "alpha"])
        print("✓ Logit model summary:")
        print(summary_logit)
    
    if 'idata_probit' in locals():
        summary_probit = az.summary(idata_probit, var_names=["beta", "alpha"])
        print("✓ Probit model summary:")
        print(summary_probit)
        
except Exception as e:
    print(f"✗ Diagnostics failed: {e}")

print("\n🎯 LOGIT/PROBIT TEST COMPLETED!") 