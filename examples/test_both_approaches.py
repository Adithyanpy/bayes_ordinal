#!/usr/bin/env python3
"""
Test script that demonstrates both constrained and unconstrained approaches like PyMCOrdinal.
"""

import numpy as np
import pymc as pm
import arviz as az
import bayes_ordinal as bo

# Set random seed
np.random.seed(42)

print("Testing both constrained and unconstrained approaches (PyMCOrdinal style)...")

# Create simple data
n_samples = 100
X = np.random.randn(n_samples, 2)
y = np.random.randint(0, 5, n_samples)  # 5 categories (K=5)
K = 5

print(f"Data: {n_samples} samples, {K} categories")

# Test 1: Unconstrained approach (PyMCOrdinal default)
print("\n1. Testing unconstrained approach (PyMCOrdinal default):")
try:
    with bo.cumulative_model(
        y=y,
        X=X,
        K=K,
        priors=bo.get_conservative_priors(model_type="ordinal", K=K),
        link="logit",
        constrained_uniform=False  # Use unconstrained approach
    ) as our_model1:
        print("✓ Our package unconstrained model created successfully")
        
        # Fit model
        our_idata1 = bo.fit_model(
            model=our_model1,
            draws=500,
            tune=200,
            chains=2,
            random_seed=42
        )
        print("✓ Our package unconstrained model fitted successfully")
        
        # Check results
        summary1 = az.summary(our_idata1)
        print("✓ Unconstrained summary generated")
        print(f"  • Beta parameters: {summary1.loc[summary1.index.str.contains('beta'), 'mean'].values}")
        print(f"  • Alpha cutpoints: {summary1.loc[summary1.index.str.contains('alpha'), 'mean'].values}")
        
except Exception as e:
    print(f"✗ Our package unconstrained failed: {e}")

# Test 2: Constrained approach (PyMCOrdinal constrainedUniform)
print("\n2. Testing constrained approach (PyMCOrdinal constrainedUniform):")
try:
    with bo.cumulative_model(
        y=y,
        X=X,
        K=K,
        priors=bo.get_conservative_priors(model_type="ordinal", K=K),
        link="logit",
        constrained_uniform=True  # Use constrained approach
    ) as our_model2:
        print("✓ Our package constrained model created successfully")
        
        # Fit model
        our_idata2 = bo.fit_model(
            model=our_model2,
            draws=500,
            tune=200,
            chains=2,
            random_seed=42
        )
        print("✓ Our package constrained model fitted successfully")
        
        # Check results
        summary2 = az.summary(our_idata2)
        print("✓ Constrained summary generated")
        print(f"  • Beta parameters: {summary2.loc[summary2.index.str.contains('beta'), 'mean'].values}")
        print(f"  • Alpha cutpoints: {summary2.loc[summary2.index.str.contains('alpha'), 'mean'].values}")
        
except Exception as e:
    print(f"✗ Our package constrained failed: {e}")

# Test 3: Direct PyMCOrdinal unconstrained approach
print("\n3. Testing direct PyMCOrdinal unconstrained approach:")
try:
    with pm.Model() as direct_model1:
        # Priors
        beta = pm.Normal("beta", 0, 1, size=2)
        sigma = pm.Exponential("sigma", 1.0)
        
        # Unconstrained cutpoints (PyMCOrdinal default)
        cutpoints = pm.Normal(
            "cutpoints",
            mu=np.linspace(0, K, K-1),
            sigma=sigma,
            shape=K-1,
            transform=pm.distributions.transforms.Ordered()
        )
        
        # Linear predictor
        mu = pm.math.dot(X, beta)
        
        # Likelihood
        y_obs = pm.OrderedLogistic("y_obs", eta=mu, cutpoints=cutpoints, observed=y)
    
    # Fit direct unconstrained model
    with direct_model1:
        direct_idata1 = pm.sample(
            draws=500,
            tune=200,
            chains=2,
            random_seed=42,
            return_inferencedata=True
        )
        print("✓ Direct PyMCOrdinal unconstrained model fitted successfully")
        
        # Check results
        summary3 = az.summary(direct_idata1)
        print("✓ Direct unconstrained summary generated")
        print(f"  • Beta parameters: {summary3.loc[summary3.index.str.contains('beta'), 'mean'].values}")
        print(f"  • Cutpoints: {summary3.loc[summary3.index.str.contains('cutpoints'), 'mean'].values}")
        
except Exception as e:
    print(f"✗ Direct PyMCOrdinal unconstrained failed: {e}")

# Test 4: Direct PyMCOrdinal constrained approach
print("\n4. Testing direct PyMCOrdinal constrained approach:")
try:
    with pm.Model() as direct_model2:
        # Priors
        beta = pm.Normal("beta", 0, 1, size=2)
        
        # Constrained cutpoints (PyMCOrdinal constrainedUniform)
        alpha_raw = pm.Dirichlet("alpha_raw", a=np.ones(K-1))
        cutpoints = pm.Deterministic("cutpoints", 
                                   pm.math.cumsum(alpha_raw) * (K-1))
        
        # Linear predictor
        mu = pm.math.dot(X, beta)
        
        # Likelihood
        y_obs = pm.OrderedLogistic("y_obs", eta=mu, cutpoints=cutpoints, observed=y)
    
    # Fit direct constrained model
    with direct_model2:
        direct_idata2 = pm.sample(
            draws=500,
            tune=200,
            chains=2,
            random_seed=42,
            return_inferencedata=True
        )
        print("✓ Direct PyMCOrdinal constrained model fitted successfully")
        
        # Check results
        summary4 = az.summary(direct_idata2)
        print("✓ Direct constrained summary generated")
        print(f"  • Beta parameters: {summary4.loc[summary4.index.str.contains('beta'), 'mean'].values}")
        print(f"  • Cutpoints: {summary4.loc[summary4.index.str.contains('cutpoints'), 'mean'].values}")
        
except Exception as e:
    print(f"✗ Direct PyMCOrdinal constrained failed: {e}")

print("\n✓ Test completed!")
print("Our package now supports both approaches like PyMCOrdinal:")
print("  • constrained_uniform=False: Unconstrained Normal with Ordered() transform")
print("  • constrained_uniform=True: Constrained Dirichlet approach") 