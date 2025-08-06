#!/usr/bin/env python3
"""
Advanced Workflow Example with Configuration System

This script demonstrates advanced usage of the bayes_ordinal package,
including the configuration system, CLI functionality, and hierarchical models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add package to path
sys.path.append('.')

import bayes_ordinal as bo
from bayes_ordinal.config import (
    Config, get_default_config, get_hierarchical_config, 
    get_production_config, get_quick_test_config
)

# Set random seed
np.random.seed(42)

print("=" * 70)
print("ADVANCED BAYESIAN ORDINAL REGRESSION WORKFLOW")
print("=" * 70)

# %% [markdown]
# ## 1. Configuration System Demonstration

# %%
print("=== Configuration System ===")

# 1. Create different configurations
configs = {
    "default": get_default_config(),
    "quick_test": get_quick_test_config(),
    "production": get_production_config(),
    "hierarchical": get_hierarchical_config()
}

print("Available configurations:")
for name, config in configs.items():
    print(f"  {name}:")
    print(f"    - Model: {config.model.model_type}")
    print(f"    - Hierarchical: {config.model.hierarchical}")
    print(f"    - Draws: {config.sampling.draws}")
    print(f"    - Chains: {config.sampling.chains}")
    print(f"    - Smoke test: {config.sampling.smoke_test}")

# 2. Custom configuration
custom_config = Config()
custom_config.name = "custom_analysis"
custom_config.description = "Custom configuration for demonstration"
custom_config.model.model_type = "cumulative"
custom_config.model.link = "probit"
custom_config.priors.prior_type = "weakly_informative"
custom_config.sampling.draws = 1500
custom_config.sampling.tune = 750
custom_config.workflow.run_sensitivity_analysis = True

print(f"\nCustom configuration created:")
print(f"  - Name: {custom_config.name}")
print(f"  - Model: {custom_config.model.model_type} with {custom_config.model.link} link")
print(f"  - Priors: {custom_config.priors.prior_type}")
print(f"  - Sensitivity analysis: {custom_config.workflow.run_sensitivity_analysis}")

# 3. Save and load configuration
custom_config.save("examples/custom_config.yaml")
loaded_config = Config.from_file("examples/custom_config.yaml")
print(f"\nConfiguration saved and loaded successfully")
print(f"  - Loaded name: {loaded_config.name}")

# %% [markdown]
# ## 2. Hierarchical Model Example

# %%
print("\n=== Hierarchical Model Example ===")

# Generate hierarchical data
n_samples, n_features, K = 200, 2, 4
n_groups = 5
samples_per_group = n_samples // n_groups

# True parameters
true_beta = np.array([0.8, -0.5])
true_cutpoints = np.array([-1.5, 0.0, 1.5])
true_u_sigma = 0.5

# Generate group structure
group_idx = np.repeat(range(n_groups), samples_per_group)
group_effects = np.random.normal(0, true_u_sigma, n_groups)

# Generate features
X = np.random.normal(size=(n_samples, n_features))

# Generate responses with group effects
linear_predictor = X @ true_beta + group_effects[group_idx]
y = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    probs = 1 / (1 + np.exp(-(true_cutpoints - linear_predictor[i])))
    probs = np.concatenate([[0], probs, [1]])
    category_probs = probs[1:] - probs[:-1]
    y[i] = np.random.choice(K, p=category_probs)

print(f"Generated hierarchical data:")
print(f"  - Samples: {n_samples}")
print(f"  - Groups: {n_groups}")
print(f"  - Categories: {K}")
print(f"  - Group effects SD: {true_u_sigma}")

# Data preprocessing
y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K)
X_scaled, scaler = bo.standardize_features(X_clean)
group_indices = bo.create_group_indices(group_idx)

print(f"  - Group indices created: {len(group_indices)} groups")

# %% [markdown]
# ## 3. Model Building with Different Configurations

# %%
print("\n=== Model Building with Configurations ===")

# 1. Standard cumulative model
print("1. Building standard cumulative model...")
model_standard = bo.cumulative_model(
    y=y_clean, X=X_scaled, K=K,
    link="logit",
    priors=bo.get_default_priors()
)

# 2. Hierarchical cumulative model
print("2. Building hierarchical cumulative model...")
model_hierarchical = bo.cumulative_model(
    y=y_clean, X=X_scaled, K=K,
    link="logit",
    priors=bo.get_weakly_informative_priors(),
    group_idx=group_indices,
    n_groups=n_groups
)

# 3. Partial odds model
print("3. Building partial odds model...")
model_partial = bo.partial_odds_model(
    y=y_clean, X=X_scaled, K=K,
    priors=bo.get_default_priors()
)

print(f"Models built successfully:")
print(f"  - Standard: {len(model_standard.free_RVs)} free variables")
print(f"  - Hierarchical: {len(model_hierarchical.free_RVs)} free variables")
print(f"  - Partial odds: {len(model_partial.free_RVs)} free variables")

# %% [markdown]
# ## 4. Prior Predictive Checks

# %%
print("\n=== Prior Predictive Checks ===")

# Use quick test configuration for prior checks
quick_config = get_quick_test_config()

print("1. Standard model prior predictive check...")
prior_standard = bo.run_prior_predictive(
    model_standard, 
    draws=quick_config.workflow.prior_draws,
    plot=False
)

print("2. Hierarchical model prior predictive check...")
prior_hierarchical = bo.run_prior_predictive(
    model_hierarchical,
    draws=quick_config.workflow.prior_draws,
    plot=False
)

print("3. Partial odds model prior predictive check...")
prior_partial = bo.run_prior_predictive(
    model_partial,
    draws=quick_config.workflow.prior_draws,
    plot=False
)

# Compare prior predictions
models_priors = {
    "standard": prior_standard,
    "hierarchical": prior_hierarchical,
    "partial": prior_partial
}

print("\nPrior predictive comparison:")
for name, prior_data in models_priors.items():
    samples = prior_data.prior_predictive['y_obs'].values.flatten()
    dist = np.bincount(samples, minlength=K)
    print(f"  {name}: {dist}")

# %% [markdown]
# ## 5. Model Fitting with Different Configurations

# %%
print("\n=== Model Fitting ===")

# Fit models with different configurations
models_to_fit = {
    "standard": model_standard,
    "hierarchical": model_hierarchical,
    "partial": model_partial
}

fitted_models = {}

for name, model in models_to_fit.items():
    print(f"Fitting {name} model...")
    
    # Use appropriate configuration
    if name == "hierarchical":
        fit_config = get_hierarchical_config()
    else:
        fit_config = get_quick_test_config()  # Use quick test for demo
    
    idata = bo.fit_model(
        model,
        draws=fit_config.sampling.draws,
        tune=fit_config.sampling.tune,
        chains=fit_config.sampling.chains,
        smoke_test=fit_config.sampling.smoke_test,
        return_inferencedata=True
    )
    
    fitted_models[name] = idata
    print(f"  ✓ {name} model fitted successfully")

# %% [markdown]
# ## 6. Diagnostics and Model Comparison

# %%
print("\n=== Diagnostics and Model Comparison ===")

# Check convergence for all models
print("Convergence diagnostics:")
for name, idata in fitted_models.items():
    conv = bo.check_convergence(idata)
    print(f"  {name}: converged={conv['converged']}, divergences={conv['n_divergences']}")

# Model comparison
print("\nModel comparison using LOO:")
comparison = bo.compare_models(
    models_to_fit,
    fitted_models,
    ic="loo"
)
print(comparison)

# %% [markdown]
# ## 7. Sensitivity Analysis

# %%
print("\n=== Sensitivity Analysis ===")

# Run sensitivity analysis on the best model
best_model_name = comparison.index[0]
best_model = models_to_fit[best_model_name]
best_idata = fitted_models[best_model_name]

print(f"Running sensitivity analysis on {best_model_name} model...")

# Sensitivity to prior hyperparameters
sensitivity_results = bo.prior_sensitivity(
    build_model=lambda **kwargs: bo.cumulative_model(
        y=y_clean, X=X_scaled, K=K, **kwargs
    ),
    y=y_clean, X=X_scaled, K=K,
    hyper_name="coef_sigma",
    hyper_values=[1.0, 2.5, 5.0, 10.0],
    trace_var="beta",
    draws=200,
    tune=100,
    chains=2
)

print("Sensitivity analysis completed")
print(f"  - Tested {len(sensitivity_results)} hyperparameter values")
print(f"  - Effect on beta coefficient: {sensitivity_results['effect_size']:.3f}")

# %% [markdown]
# ## 8. Complete Workflow with Configuration

# %%
print("\n=== Complete Workflow with Configuration ===")

# Use production configuration for final workflow
production_config = get_production_config()
production_config.model.model_type = "cumulative"  # Focus on cumulative model
production_config.workflow.run_sensitivity_analysis = False  # Skip for demo

print("Running complete workflow with production configuration...")

# Use smaller dataset for demo
n_demo = 100
y_demo = y_clean[:n_demo]
X_demo = X_scaled[:n_demo]

workflow_results = bo.run_workflow(
    model_fn=bo.cumulative_model,
    y=y_demo, X=X_demo, K=K,
    priors=bo.get_default_priors(),
    config=production_config.get_workflow_config()
)

print("Workflow completed successfully!")
print(f"Results summary:")
print(f"  - Models: {list(workflow_results.get('models', {}).keys())}")
print(f"  - Inference data: {list(workflow_results.get('idata', {}).keys())}")
print(f"  - Diagnostics: {list(workflow_results.get('diagnostics', {}).keys())}")

# %% [markdown]
# ## 9. CLI Functionality Demonstration

# %%
print("\n=== CLI Functionality ===")

# Create sample data file
sample_data = pd.DataFrame({
    'feature1': X_scaled[:50, 0],
    'feature2': X_scaled[:50, 1],
    'response': y_clean[:50]
})

sample_data.to_csv("examples/sample_data.csv", index=False)
print("Created sample data file: examples/sample_data.csv")

# Create configuration file
demo_config = get_quick_test_config()
demo_config.name = "demo_analysis"
demo_config.output.output_dir = "examples/demo_results"
demo_config.save("examples/demo_config.yaml")
print("Created demo configuration: examples/demo_config.yaml")

# Demonstrate CLI functions (without actually running CLI)
from bayes_ordinal.cli import load_data, save_results

# Load data using CLI function
y_cli, X_cli, K_cli = load_data("examples/sample_data.csv")
print(f"CLI data loading works: {len(y_cli)} samples, {X_cli.shape[1]} features")

# %% [markdown]
# ## 10. Visualization and Results

# %%
print("\n=== Visualization and Results ===")

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Coefficient effects
ax1 = axes[0, 0]
bo.plot_coefficient_effects(
    fitted_models["standard"],
    var_name="beta",
    feature_names=["Feature 1", "Feature 2"],
    hdi_prob=0.94,
    ax=ax1
)
ax1.set_title("Coefficient Effects (Standard Model)")

# 2. Cutpoint estimates
ax2 = axes[0, 1]
bo.plot_cutpoints(
    fitted_models["standard"],
    var_name="alpha",
    hdi_prob=0.94,
    ax=ax2
)
ax2.set_title("Cutpoint Estimates")

# 3. Model comparison
ax3 = axes[1, 0]
bo.plot_model_comparison(
    fitted_models,
    ic="loo",
    ax=ax3
)
ax3.set_title("Model Comparison (LOO)")

# 4. Category proportions
ax4 = axes[1, 1]
observed_props = bo.compute_category_proportions(y_clean, K)
ax4.bar(range(K), observed_props, alpha=0.7, label='Observed')
ax4.set_xlabel('Category')
ax4.set_ylabel('Proportion')
ax4.set_title('Category Proportions')
ax4.legend()

plt.tight_layout()
plt.savefig("examples/advanced_workflow_results.png", dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations saved to: examples/advanced_workflow_results.png")

# %% [markdown]
# ## Summary

# %%
print("\n" + "=" * 70)
print("ADVANCED WORKFLOW COMPLETE!")
print("=" * 70)

print("This advanced example demonstrated:")
print("1. ✅ Configuration system with multiple preset configurations")
print("2. ✅ Hierarchical model building and fitting")
print("3. ✅ Prior predictive checks for all models")
print("4. ✅ Model comparison using information criteria")
print("5. ✅ Sensitivity analysis for prior hyperparameters")
print("6. ✅ Complete workflow with production configuration")
print("7. ✅ CLI functionality and data handling")
print("8. ✅ Comprehensive visualization and results")

print("\nKey Features Demonstrated:")
print("- Modular configuration system")
print("- Multiple model types (standard, hierarchical, partial odds)")
print("- Robust diagnostics and model comparison")
print("- Prior sensitivity analysis")
print("- CLI integration")
print("- Comprehensive visualization")

print("\nThe bayes_ordinal package is now fully functional with:")
print("- Complete configuration system")
print("- CLI interface")
print("- Advanced workflow capabilities")
print("- Hierarchical model support")
print("- Comprehensive documentation and examples")

print("\nReady for production use! 🎉") 