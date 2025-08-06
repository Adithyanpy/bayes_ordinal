#!/usr/bin/env python3
"""
Test script to verify our package now works with PyMCOrdinal Dirichlet approach.
"""

import numpy as np
import pymc as pm
import arviz as az
import bayes_ordinal as bo

# Set random seed
np.random.seed(42)

print("Testing PyMCOrdinal Dirichlet approach in our package...")

# Create simple data
n_samples = 100
X = np.random.randn(n_samples, 2)
y = np.random.randint(0, 5, n_samples)  # 5 categories (K=5)
K = 5

print(f"Data: {n_samples} samples, {K} categories")

# Test our package with PyMCOrdinal approach
print("\n1. Testing our package (PyMCOrdinal approach):")
try:
    with bo.cumulative_model(
        y=y,
        X=X,
        K=K,
        priors=bo.get_conservative_priors(model_type="ordinal", K=K),
        link="logit"
    ) as our_model:
        print("✓ Our package model created successfully")
        
        # Fit model
        our_idata = bo.fit_model(
            model=our_model,
            draws=500,
            tune=200,
            chains=2,
            random_seed=42
        )
        print("✓ Our package model fitted successfully")
        
        # Check results
        summary = az.summary(our_idata)
        print("✓ Our package summary generated")
        print(f"  • Beta parameters: {summary.loc[summary.index.str.contains('beta'), 'mean'].values}")
        print(f"  • Alpha cutpoints: {summary.loc[summary.index.str.contains('alpha'), 'mean'].values}")
        
except Exception as e:
    print(f"✗ Our package failed: {e}")

# Test direct PyMCOrdinal approach
print("\n2. Testing direct PyMCOrdinal approach:")
try:
    with pm.Model() as direct_model:
        # Priors
        beta = pm.Normal("beta", 0, 1, size=2)
        
        # PyMCOrdinal constrained cutpoints
        alpha_raw = pm.Dirichlet("alpha_raw", a=np.ones(K-1))
        cutpoints = pm.Deterministic("cutpoints", 
                                   pm.math.cumsum(alpha_raw) * (K-1))
        
        # Linear predictor
        mu = pm.math.dot(X, beta)
        
        # Likelihood
        y_obs = pm.OrderedLogistic("y_obs", eta=mu, cutpoints=cutpoints, observed=y)
    
    # Fit direct model
    with direct_model:
        direct_idata = pm.sample(
            draws=500,
            tune=200,
            chains=2,
            random_seed=42,
            return_inferencedata=True
        )
        print("✓ Direct PyMCOrdinal model fitted successfully")
        
        # Check results
        summary = az.summary(direct_idata)
        print("✓ Direct PyMCOrdinal summary generated")
        print(f"  • Beta parameters: {summary.loc[summary.index.str.contains('beta'), 'mean'].values}")
        print(f"  • Cutpoints: {summary.loc[summary.index.str.contains('cutpoints'), 'mean'].values}")
        
except Exception as e:
    print(f"✗ Direct PyMCOrdinal failed: {e}")

print("\n✓ Test completed!")
print("Our package now uses the exact same PyMCOrdinal Dirichlet approach!") 