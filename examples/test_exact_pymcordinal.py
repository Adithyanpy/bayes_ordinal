#!/usr/bin/env python3
"""
Test script that exactly matches PyMCOrdinal documentation approach.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pandas as pd

# Set random seed
np.random.seed(42)

print("Testing exact PyMCOrdinal approach...")

# Create data exactly like PyMCOrdinal
np.random.seed(42)
n_samples = 1000
salary = np.random.normal(50, 15, n_samples)
work_sat = np.random.normal(3.5, 1.0, n_samples)
work_from_home = np.random.binomial(1, 0.3, n_samples)

latent_rating = 5 + 0.1 * salary + 0.5 * work_sat + 0.3 * work_from_home + np.random.normal(0, 1, n_samples)
explicit_rating = np.clip(np.round(latent_rating), 1, 10).astype(int)

df = pd.DataFrame({
    'salary': salary,
    'work_sat': work_sat,
    'work_from_home': work_from_home,
    'latent_rating': latent_rating,
    'explicit_rating': explicit_rating
})

K = 10  # 10 categories (1-10)

print(f"Data: {n_samples} samples, {K} categories")

# PyMCOrdinal constrainedUniform function (exact copy)
def constrainedUniform(N, min=0, max=1):
    return pm.Deterministic(
        "cutpoints",
        pt.concatenate([
            np.ones(1) * min,
            pt.extra_ops.cumsum(pm.Dirichlet("cuts_unknown", a=np.ones(N - 2))) * (max - min) + min,
        ])
    )

# PyMCOrdinal make_model function (exact copy)
def make_model(priors, model_spec=1, constrained_uniform=False, logit=True):
    with pm.Model() as model:
        if constrained_uniform:
            cutpoints = constrainedUniform(K, 0, K)
        else:
            sigma = pm.Exponential("sigma", priors["sigma"])
            cutpoints = pm.Normal(
                "cutpoints",
                mu=priors["mu"],
                sigma=sigma,
                transform=pm.distributions.transforms.univariate_ordered,
            )

        if model_spec == 1:
            beta = pm.Normal("beta", priors["beta"][0], priors["beta"][1], size=1)
            mu = pm.Deterministic("mu", beta[0] * df.salary)
        elif model_spec == 2:
            beta = pm.Normal("beta", priors["beta"][0], priors["beta"][1], size=2)
            mu = pm.Deterministic("mu", beta[0] * df.salary + beta[1] * df.work_sat)
        elif model_spec == 3:
            beta = pm.Normal("beta", priors["beta"][0], priors["beta"][1], size=3)
            mu = pm.Deterministic(
                "mu", beta[0] * df.salary + beta[1] * df.work_sat + beta[2] * df.work_from_home
            )
        if logit:
            y_ = pm.OrderedLogistic("y", cutpoints=cutpoints, eta=mu, observed=df.explicit_rating)
        else:
            y_ = pm.OrderedProbit("y", cutpoints=cutpoints, eta=mu, observed=df.explicit_rating)
        
        return model

# Test the exact PyMCOrdinal approach
print("\nTesting exact PyMCOrdinal constrainedUniform approach:")
try:
    priors = {"sigma": 1, "beta": [0, 1], "mu": np.linspace(0, K, K - 1)}
    
    # Test constrained uniform (Model 4 from PyMCOrdinal)
    model4 = make_model(priors, model_spec=3, constrained_uniform=True, logit=True)
    
    with model4:
        idata4 = pm.sample(
            draws=500,
            tune=200,
            chains=2,
            random_seed=42,
            return_inferencedata=True
        )
    
    print("✓ Exact PyMCOrdinal constrainedUniform model fitted successfully")
    
    # Check results
    summary = az.summary(idata4)
    print("✓ Summary generated")
    print(f"  • Beta parameters: {summary.loc[summary.index.str.contains('beta'), 'mean'].values}")
    print(f"  • Cutpoints: {summary.loc[summary.index.str.contains('cutpoints'), 'mean'].values}")
    
except Exception as e:
    print(f"✗ Exact PyMCOrdinal failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Test completed!") 