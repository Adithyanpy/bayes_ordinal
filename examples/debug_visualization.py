#!/usr/bin/env python3
"""
Debug script to identify visualization issues.
"""

import numpy as np
import pymc as pm
import arviz as az
import bayes_ordinal as bo

print("🔍 DEBUGGING VISUALIZATION ISSUES")
print("=" * 40)

# Create simple test data
np.random.seed(42)
n_samples = 50
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

# Test each visualization function
print("\n🧪 TESTING VISUALIZATION FUNCTIONS")

# Test 1: plot_forest
print("\n1️⃣ Testing plot_forest...")
try:
    bo.plot_forest(idata, var_name="beta", feature_names=["Feature 1", "Feature 2"])
    print("✓ plot_forest works")
except Exception as e:
    print(f"✗ plot_forest failed: {e}")

# Test 2: plot_coefficient_effects
print("\n2️⃣ Testing plot_coefficient_effects...")
try:
    bo.plot_coefficient_effects(idata, var_name="beta", feature_names=["Feature 1", "Feature 2"])
    print("✓ plot_coefficient_effects works")
except Exception as e:
    print(f"✗ plot_coefficient_effects failed: {e}")

# Test 3: plot_cutpoints
print("\n3️⃣ Testing plot_cutpoints...")
try:
    bo.plot_cutpoints(idata)
    print("✓ plot_cutpoints works")
except Exception as e:
    print(f"✗ plot_cutpoints failed: {e}")

# Test 4: plot_category_probabilities
print("\n4️⃣ Testing plot_category_probabilities...")
try:
    bo.plot_category_probabilities(idata, model, X[:5])
    print("✓ plot_category_probabilities works")
except Exception as e:
    print(f"✗ plot_category_probabilities failed: {e}")

# Test 5: plot_model_structure
print("\n5️⃣ Testing plot_model_structure...")
try:
    bo.plot_model_structure(model)
    print("✓ plot_model_structure works")
except Exception as e:
    print(f"✗ plot_model_structure failed: {e}")

print("\n✅ DEBUGGING COMPLETED") 