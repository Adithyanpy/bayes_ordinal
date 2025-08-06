#!/usr/bin/env python3
"""
Comprehensive test script for Bayesian ordinal regression package.

This script demonstrates all the functionality of the bayes_ordinal package
in a proper workflow manner with detailed documentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import bayes_ordinal as bo

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("COMPREHENSIVE BAYESIAN ORDINAL REGRESSION WORKFLOW")
print("=" * 60)
print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")
print(f"Bayes Ordinal version: {bo.__version__}")
print()

# ============================================================================
# 1. DATA GENERATION AND PREPROCESSING
# ============================================================================

print("1. DATA GENERATION AND PREPROCESSING")
print("-" * 40)

# Generate synthetic ordinal data
n_samples, n_features, K = 200, 4, 5

# True parameters for data generation
true_beta = np.array([0.8, -0.5, 0.3, -0.2])
true_cutpoints = np.array([-2.0, -0.8, 0.2, 1.5])

# Generate features (some continuous, some categorical)
X_continuous = np.random.normal(size=(n_samples, 2))
X_categorical = np.random.choice(['A', 'B', 'C'], size=(n_samples, 2))

# Combine features
X_df = pd.DataFrame({
    'age': X_continuous[:, 0],
    'income': X_continuous[:, 1],
    'region': X_categorical[:, 0],
    'education': X_categorical[:, 1]
})

# Generate ordinal responses using cumulative model
linear_predictor = X_continuous @ true_beta[:2]  # Only use continuous features for generation
y = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    # Cumulative probabilities using logit link
    probs = 1 / (1 + np.exp(-(true_cutpoints - linear_predictor[i])))
    probs = np.concatenate([[0], probs, [1]])
    
    # Sample category
    category_probs = probs[1:] - probs[:-1]
    y[i] = np.random.choice(K, p=category_probs)

print(f"Generated {n_samples} samples with {K} categories")
print(f"Category distribution: {np.bincount(y, minlength=K)}")
print(f"Feature matrix shape: {X_df.shape}")

# Data preprocessing using our utility functions
print("\nData preprocessing:")
# 1. Validate ordinal data
y_clean, X_clean, K = bo.validate_ordinal_data(y, X_df, K)
print(f"✓ Data validation passed")

# 2. Encode categorical features
X_encoded, encoders = bo.encode_categorical_features(X_clean, ['region', 'education'])
print(f"✓ Categorical features encoded")

# 3. Standardize continuous features
X_scaled, scaler = bo.standardize_features(X_encoded)
print(f"✓ Features standardized")

# 4. Compute category proportions
proportions = bo.compute_category_proportions(y_clean, K)
print(f"✓ Category proportions computed: {proportions}")

# ============================================================================
# 2. PRIOR SPECIFICATION
# ============================================================================

print("\n\n2. PRIOR SPECIFICATION")
print("-" * 40)

# 1. Default priors
default_priors = bo.get_default_priors()
print("1. Default Priors:")
for key, value in default_priors.items():
    print(f"   {key}: {value}")

# 2. Weakly informative priors
weak_priors = bo.get_weakly_informative_priors()
print("\n2. Weakly Informative Priors:")
for key, value in weak_priors.items():
    print(f"   {key}: {value}")

# 3. Custom informative priors
custom_priors = bo.get_informative_priors(
    coef_means=np.array([0.5, -0.3, 0.0, 0.0]),
    coef_sds=np.array([1.0, 1.0, 0.5, 0.5]),
    cut_means=np.array([-1.5, -0.5, 0.5, 1.5]),
    cut_sds=np.array([1.0, 1.0, 1.0, 1.0])
)
print("\n3. Custom Informative Priors:")
for key, value in custom_priors.items():
    if isinstance(value, np.ndarray):
        print(f"   {key}: {value}")
    else:
        print(f"   {key}: {value}")

# 4. Validate priors
validated_priors = bo.validate_priors(default_priors, model_type="cumulative")
print(f"\n4. Prior validation passed: {len(validated_priors)} parameters validated")

# ============================================================================
# 3. MODEL BUILDING
# ============================================================================

print("\n\n3. MODEL BUILDING")
print("-" * 40)

# 1. Cumulative model with logit link
model_cum_logit = bo.cumulative_model(
    y=y_clean, X=X_scaled, K=K,
    link="logit",
    priors=default_priors
)
print("1. Cumulative model (logit link) created")
print(f"   - Free variables: {[var.name for var in model_cum_logit.free_RVs]}")

# 2. Cumulative model with probit link
model_cum_probit = bo.cumulative_model(
    y=y_clean, X=X_scaled, K=K,
    link="probit",
    priors=default_priors
)
print("2. Cumulative model (probit link) created")

# 3. Partial odds model
model_partial = bo.partial_odds_model(
    y=y_clean, X=X_scaled, K=K,
    priors=default_priors
)
print("3. Partial odds model created")
print(f"   - Free variables: {[var.name for var in model_partial.free_RVs]}")

# 4. Check default initial values
print("4. Default initial values:")
print(f"   Cumulative model: {list(model_cum_logit._default_initvals.keys())}")
print(f"   Partial model: {list(model_partial._default_initvals.keys())}")

# ============================================================================
# 4. PRIOR PREDICTIVE CHECKS
# ============================================================================

print("\n\n4. PRIOR PREDICTIVE CHECKS")
print("-" * 40)

# 1. Cumulative model (logit)
print("1. Running prior predictive check for cumulative model (logit)...")
prior_idata_cum_logit = bo.run_prior_predictive(
    model_cum_logit, draws=100, plot=False
)

# 2. Partial odds model
print("2. Running prior predictive check for partial odds model...")
prior_idata_partial = bo.run_prior_predictive(
    model_partial, draws=100, plot=False
)

# 3. Compare prior predictions
cum_prior_samples = prior_idata_cum_logit.prior_predictive['y_obs'].values.flatten()
partial_prior_samples = prior_idata_partial.prior_predictive['y_obs'].values.flatten()

print("3. Comparing prior predictions:")
print(f"   Cumulative model - Category distribution: {np.bincount(cum_prior_samples, minlength=K)}")
print(f"   Partial odds model - Category distribution: {np.bincount(partial_prior_samples, minlength=K)}")
print(f"   Observed data - Category distribution: {np.bincount(y_clean, minlength=K)}")

# ============================================================================
# 5. MODEL FITTING
# ============================================================================

print("\n\n5. MODEL FITTING")
print("-" * 40)

# 1. Fit cumulative model (logit)
print("1. Fitting cumulative model (logit)...")
idata_cum_logit = bo.fit_model(
    model_cum_logit,
    draws=500,
    tune=200,
    chains=2,
    smoke_test=False,
    return_inferencedata=True
)
print(f"   ✓ Cumulative model fitted successfully")
print(f"   - Posterior samples: {idata_cum_logit.posterior.dims}")

# 2. Fit partial odds model
print("2. Fitting partial odds model...")
idata_partial = bo.fit_model(
    model_partial,
    draws=500,
    tune=200,
    chains=2,
    smoke_test=False,
    return_inferencedata=True
)
print(f"   ✓ Partial odds model fitted successfully")
print(f"   - Posterior samples: {idata_partial.posterior.dims}")

# ============================================================================
# 6. CONVERGENCE DIAGNOSTICS
# ============================================================================

print("\n\n6. CONVERGENCE DIAGNOSTICS")
print("-" * 40)

# 1. Check convergence for cumulative model
print("1. Cumulative model convergence:")
conv_cum = bo.check_convergence(idata_cum_logit)
print(f"   - Converged: {conv_cum['converged']}")
print(f"   - Divergences: {conv_cum['n_divergences']}")
print(f"   - Bad R-hat: {conv_cum['n_bad_rhat']}")
print(f"   - Bad ESS: {conv_cum['n_bad_ess']}")

# 2. Check convergence for partial odds model
print("2. Partial odds model convergence:")
conv_partial = bo.check_convergence(idata_partial)
print(f"   - Converged: {conv_partial['converged']}")
print(f"   - Divergences: {conv_partial['n_divergences']}")
print(f"   - Bad R-hat: {conv_partial['n_bad_rhat']}")
print(f"   - Bad ESS: {conv_partial['n_bad_ess']}")

# 3. Detailed diagnostics
print("3. Detailed diagnostics for cumulative model:")
diag_cum = bo.summarize_diagnostics(idata_cum_logit, var_names=["beta", "alpha"])
print(diag_cum[['mean', 'sd', 'r_hat', 'ess_bulk', 'n_divergences']])

# ============================================================================
# 7. POSTERIOR PREDICTIVE CHECKS
# ============================================================================

print("\n\n7. POSTERIOR PREDICTIVE CHECKS")
print("-" * 40)

# 1. Cumulative model PPC
print("1. Posterior predictive check for cumulative model:")
ppc_cum = bo.run_posterior_predictive(
    model_cum_logit, idata_cum_logit,
    kind="proportions"
)

# 2. Partial odds model PPC
print("2. Posterior predictive check for partial odds model:")
ppc_partial = bo.run_posterior_predictive(
    model_partial, idata_partial,
    kind="proportions"
)

print("   ✓ Posterior predictive checks completed")

# ============================================================================
# 8. MODEL COMPARISON
# ============================================================================

print("\n\n8. MODEL COMPARISON")
print("-" * 40)

# 1. Compare using LOO
print("1. Model comparison using LOO:")
comparison_loo = bo.compare_models(
    {"cumulative": model_cum_logit, "partial": model_partial},
    {"cumulative": idata_cum_logit, "partial": idata_partial},
    ic="loo"
)
print(comparison_loo)

# 2. Compare using WAIC
print("\n2. Model comparison using WAIC:")
comparison_waic = bo.compare_models(
    {"cumulative": model_cum_logit, "partial": model_partial},
    {"cumulative": idata_cum_logit, "partial": idata_partial},
    ic="waic"
)
print(comparison_waic)

# ============================================================================
# 9. SENSITIVITY ANALYSIS
# ============================================================================

print("\n\n9. SENSITIVITY ANALYSIS")
print("-" * 40)

# 1. Prior sensitivity for coefficient standard deviation
print("1. Prior sensitivity analysis for coefficient standard deviation:")
bo.prior_sensitivity(
    bo.cumulative_model,
    y=y_clean, X=X_scaled, K=K,
    hyper_name="coef_sigma",
    hyper_values=[1.0, 2.5, 5.0],
    trace_var="beta",
    draws=200,
    tune=100,
    chains=2
)

# 2. Influence diagnostics
print("2. Influence diagnostics for cumulative model:")
bo.plot_influential(idata_cum_logit, threshold=0.7)

# ============================================================================
# 10. VISUALIZATION
# ============================================================================

print("\n\n10. VISUALIZATION")
print("-" * 40)

# 1. Coefficient effects
print("1. Coefficient effects plot:")
feature_names = ['Age', 'Income', 'Region', 'Education']
bo.plot_coefficient_effects(
    idata_cum_logit,
    var_name="beta",
    feature_names=feature_names,
    hdi_prob=0.94
)

# 2. Cutpoint estimates
print("2. Cutpoint estimates plot:")
bo.plot_cutpoints(
    idata_cum_logit,
    var_name="alpha",
    hdi_prob=0.94
)

# 3. Prior vs posterior comparison
print("3. Prior vs posterior comparison for first coefficient:")
prior_beta_samples = prior_idata_cum_logit.prior['beta'].values[:, :, 0].flatten()
bo.plot_prior_posterior(
    idata_cum_logit,
    var_name="beta",
    prior_samples=prior_beta_samples
)

# ============================================================================
# 11. HIERARCHICAL MODEL EXAMPLE
# ============================================================================

print("\n\n11. HIERARCHICAL MODEL EXAMPLE")
print("-" * 40)

# Create group structure
n_groups = 5
group_sizes = np.random.randint(20, 40, size=n_groups)
group_idx = np.concatenate([np.repeat(i, size) for i, size in enumerate(group_sizes)])
n_total = len(group_idx)

# Generate hierarchical data
X_hier = np.random.normal(size=(n_total, 2))
group_effects = np.random.normal(0, 0.5, size=n_groups)
linear_predictor_hier = X_hier @ np.array([0.5, -0.3]) + group_effects[group_idx]

# Generate ordinal responses
y_hier = np.zeros(n_total, dtype=int)
for i in range(n_total):
    probs = 1 / (1 + np.exp(-(true_cutpoints - linear_predictor_hier[i])))
    probs = np.concatenate([[0], probs, [1]])
    category_probs = probs[1:] - probs[:-1]
    y_hier[i] = np.random.choice(K, p=category_probs)

print(f"Generated hierarchical data with {n_groups} groups and {n_total} total observations")

# Create group indices
group_idx_clean, n_groups_clean, group_map = bo.create_group_indices(group_idx)
print(f"Group structure: {n_groups_clean} groups")

# Fit hierarchical model
model_hier = bo.cumulative_model(
    y=y_hier, X=X_hier, K=K,
    group_idx=group_idx_clean,
    n_groups=n_groups_clean,
    priors=default_priors
)

print("Fitting hierarchical model...")
idata_hier = bo.fit_model(
    model_hier,
    draws=500,
    tune=200,
    chains=2,
    smoke_test=False
)

# Plot group effects
print("Plotting group-level effects:")
bo.plot_group_forest(idata_hier, var_name="u", hdi_prob=0.9)

# ============================================================================
# 12. COMPLETE WORKFLOW
# ============================================================================

print("\n\n12. COMPLETE WORKFLOW")
print("-" * 40)

# Use a smaller dataset for the complete workflow
n_small = 100
X_small = X_scaled[:n_small]
y_small = y_clean[:n_small]

# Run complete workflow
results = bo.run_workflow(
    model_fn={
        "cumulative": bo.cumulative_model,
        "partial": bo.partial_odds_model
    },
    y=y_small, X=X_small, K=K,
    config={
        "draws": 300,
        "tune": 150,
        "chains": 2,
        "run_prior": True,
        "run_ppc": True,
        "run_diagnostics": True,
        "run_cv": True,
        "smoke_test": False
    }
)

print("Workflow completed successfully!")
print(f"Results summary:")
print(f"- Models built: {list(results['models'].keys())}")
print(f"- Inference data: {list(results['idata'].keys())}")
print(f"- Diagnostics: {list(results['diagnostics'].keys())}")
print(f"- Cross-validation: {type(results['cv']).__name__}")

# Show model comparison results
if 'cv' in results and hasattr(results['cv'], 'index'):
    print(f"Model comparison results:")
    print(results['cv'])
    
    best_model = results['cv'].index[0]
    print(f"Best model: {best_model}")
    print(f"LOO difference: {results['cv'].iloc[0]['elpd_diff']:.3f}")

# ============================================================================
# SUMMARY AND CONCLUSIONS
# ============================================================================

print("\n\n" + "=" * 60)
print("SUMMARY AND CONCLUSIONS")
print("=" * 60)

print("This script has demonstrated the complete functionality of the bayes_ordinal package:")
print()
print("Key Features Demonstrated:")
print("1. Data Preprocessing: Validation, encoding, and standardization utilities")
print("2. Prior Specification: Default, weakly informative, and custom priors")
print("3. Model Building: Cumulative and partial odds models with different link functions")
print("4. Prior Predictive Checks: Validating prior assumptions")
print("5. Model Fitting: Robust MCMC sampling with fallback strategies")
print("6. Convergence Diagnostics: Comprehensive MCMC diagnostics")
print("7. Posterior Predictive Checks: Model validation with multiple visualization options")
print("8. Model Comparison: LOO and WAIC-based model selection")
print("9. Sensitivity Analysis: Prior sensitivity and influence diagnostics")
print("10. Visualization: Specialized plots for ordinal regression")
print("11. Hierarchical Models: Group-level varying intercepts")
print("12. Automated Workflow: Complete end-to-end analysis")
print()
print("Best Practices Followed:")
print("- Bayesian Workflow: Following the principles from Gelman et al. and McElreath")
print("- Robust Diagnostics: Comprehensive convergence and model checking")
print("- Multiple Models: Comparing different specifications")
print("- Prior Sensitivity: Checking robustness to prior choices")
print("- Reproducibility: Setting random seeds and documenting all steps")
print()
print("Package Strengths:")
print("- Comprehensive: Covers the full Bayesian analysis pipeline")
print("- User-Friendly: High-level functions with sensible defaults")
print("- Robust: Built-in error handling and fallback strategies")
print("- Extensible: Modular design for easy customization")
print("- Well-Documented: Clear examples and documentation")
print()
print("The package is now ready for use in real-world ordinal regression analyses!")
print("=" * 60) 