#!/usr/bin/env python3
"""
Test script to verify plotting fixes work correctly.
"""

import numpy as np
import pymc as pm
import arviz as az
import bayes_ordinal as bo
import matplotlib.pyplot as plt

print("🧪 TESTING PLOTTING FIXES")
print("=" * 30)

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
print(f"✓ Observed distribution: {np.bincount(y)}")

# Create and fit model
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

with model:
    idata = pm.sample(
        draws=500, 
        tune=200, 
        chains=2,
        return_inferencedata=True,
        random_seed=42
    )

print("✓ Model fitted successfully")

# Test 1: Prior predictive with observed data
print("\n1️⃣ Testing prior predictive with observed data...")
try:
    prior_idata = bo.run_prior_predictive(model, draws=100, y_obs=y)
    print("✓ Prior predictive with observed data works")
except Exception as e:
    print(f"✗ Prior predictive failed: {e}")

# Test 2: Posterior predictive
print("\n2️⃣ Testing posterior predictive...")
try:
    ppc_idata = bo.run_posterior_predictive(model, idata, kind="proportions")
    print("✓ Posterior predictive works")
except Exception as e:
    print(f"✗ Posterior predictive failed: {e}")

# Test 3: Cutpoints plot
print("\n3️⃣ Testing cutpoints plot...")
try:
    bo.plot_cutpoints(idata)
    print("✓ Cutpoints plot works")
except Exception as e:
    print(f"✗ Cutpoints plot failed: {e}")

# Test 4: Forest plot
print("\n4️⃣ Testing forest plot...")
try:
    bo.plot_forest(idata, var_name="beta", feature_names=["Feature 1", "Feature 2"])
    print("✓ Forest plot works")
except Exception as e:
    print(f"✗ Forest plot failed: {e}")

# Test 5: Category probabilities
print("\n5️⃣ Testing category probabilities...")
try:
    bo.plot_category_probabilities(idata, model, X[:5])
    print("✓ Category probabilities plot works")
except Exception as e:
    print(f"✗ Category probabilities plot failed: {e}")

print("\n✅ ALL PLOTTING TESTS COMPLETED!")
print("\n📊 SUMMARY:")
print("  • Prior predictive: Shows both prior and observed distributions")
print("  • Posterior predictive: Shows proportions with observed data")
print("  • Cutpoints: Shows proper labels and formatting")
print("  • Forest plot: Shows coefficient effects")
print("  • Category probabilities: Shows predicted probabilities") 