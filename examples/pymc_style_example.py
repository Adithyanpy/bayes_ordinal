#!/usr/bin/env python3
"""
PyMC-Style Bayesian Ordinal Regression Example

This example demonstrates how to use the bayes_ordinal package following
PyMC best practices and the official Ordered Categories documentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import bayes_ordinal as bo

# Set up plotting style
plt.style.use("default")
sns.set_palette("viridis")

def main():
    """Run PyMC-style ordinal regression analysis."""
    
    print("PyMC-Style Bayesian Ordinal Regression")
    print("======================================")
    
    # 1. Data Generation (following PyMC documentation style)
    print("\n1. Generating synthetic data...")
    
    np.random.seed(42)
    N_SAMPLES = 200
    N_FEATURES = 3
    N_RESPONSE_CLASSES = 5
    
    # Generate predictors
    X = np.random.normal(size=(N_SAMPLES, N_FEATURES))
    
    # True parameters (following PyMC documentation)
    true_beta = np.array([0.5, -0.3, 0.8])
    true_cutpoints = np.array([-2.0, -1.0, 0.0, 1.0])
    
    # Generate ordinal responses
    linear_predictor = X @ true_beta
    y = np.zeros(N_SAMPLES, dtype=int)
    
    for i in range(N_SAMPLES):
        # Cumulative probabilities
        probs = 1 / (1 + np.exp(-(true_cutpoints - linear_predictor[i])))
        probs = np.concatenate([[0], probs, [1]])
        
        # Sample category
        category_probs = probs[1:] - probs[:-1]
        y[i] = np.random.choice(N_RESPONSE_CLASSES, p=category_probs)
    
    print(f"   Generated {N_SAMPLES} samples with {N_RESPONSE_CLASSES} categories")
    print(f"   Category distribution: {np.bincount(y, minlength=N_RESPONSE_CLASSES)}")
    
    # 2. Data Validation (PyMC best practices)
    print("\n2. Validating data...")
    
    validation_results = bo.validate_ordinal_model(y, X, N_RESPONSE_CLASSES)
    
    if validation_results["valid"]:
        print("   ✓ Data validation passed")
    else:
        print("   ✗ Data validation failed")
    
    if validation_results["warnings"]:
        print("   Warnings:")
        for warning in validation_results["warnings"]:
            print(f"     - {warning}")
    
    if validation_results["recommendations"]:
        print("   Recommendations:")
        for rec in validation_results["recommendations"]:
            print(f"     - {rec}")
    
    # 3. Data Preprocessing
    print("\n3. Preprocessing data...")
    
    # Standardize predictors (PyMC recommendation)
    X_centered = X - X.mean(axis=0)
    X_scaled = X_centered / X.std(axis=0)
    
    print("   ✓ Predictors standardized")
    
    # 4. Model Specification (following PyMC documentation)
    print("\n4. Specifying model...")
    
    # Use PyMC-style conservative priors
    priors = bo.get_conservative_priors(K=N_RESPONSE_CLASSES)
    
    print("   Prior specifications:")
    print(f"     - Cutpoints: Normal(0, 1) with univariate_ordered transform")
    print(f"     - Coefficients: Normal(0, {priors['coef_sigma']})")
    print(f"     - Initialization: arange(K-1) - (K-1)/2")
    
    # 5. Model Building
    print("\n5. Building model...")
    
    model = bo.cumulative_model(
        y=y,
        X=X_scaled,
        K=N_RESPONSE_CLASSES,
        priors=priors,
        link="logit",
        constrained_uniform=False,  # Use PyMC's unconstrained approach
        model_name="ordinal_regression"
    )
    
    print(f"   ✓ Model built with {len(model.free_RVs)} parameters")
    print(f"   Model variables: {[var.name for var in model.free_RVs]}")
    
    # 6. Prior Predictive Checks
    print("\n6. Running prior predictive checks...")
    
    with model:
        prior_idata = bo.run_prior_predictive(model, draws=200, plot=False)
    
    # Check prior predictions
    prior_y = prior_idata.prior_predictive["y_obs"].values.flatten()
    print(f"   Prior predictive range: {prior_y.min()}-{prior_y.max()}")
    print(f"   Prior predictive mean: {prior_y.mean():.2f}")
    
    # 7. Model Fitting
    print("\n7. Fitting model...")
    
    with model:
        idata = pm.sample(
            draws=1000,
            tune=500,
            chains=4,
            return_inferencedata=True,
            random_seed=42
        )
    
    print("   ✓ Model fitted successfully")
    
    # 8. Model Diagnostics
    print("\n8. Running diagnostics...")
    
    # Create comprehensive model summary
    model_summary = bo.create_model_summary(idata, "ordinal_regression")
    
    print(f"   Sampling info:")
    print(f"     - Chains: {model_summary['sampling_info']['n_chains']}")
    print(f"     - Draws: {model_summary['sampling_info']['n_draws']}")
    print(f"     - Parameters: {model_summary['sampling_info']['n_parameters']}")
    
    print(f"   Diagnostics:")
    print(f"     - Max R-hat: {model_summary['diagnostics']['max_rhat']:.3f}")
    print(f"     - Min ESS: {model_summary['diagnostics']['min_ess']:.0f}")
    print(f"     - Divergences: {model_summary['diagnostics']['n_divergences']}")
    print(f"     - Converged: {model_summary['diagnostics']['converged']}")
    
    if model_summary["recommendations"]:
        print("   Recommendations:")
        for rec in model_summary["recommendations"]:
            print(f"     - {rec}")
    
    # 9. Posterior Analysis
    print("\n9. Analyzing posterior...")
    
    # Parameter summary
    print("   Parameter estimates:")
    for var_name, stats in model_summary["parameter_summary"].items():
        print(f"     {var_name}: {stats['mean']:.3f} [{stats['q2_5']:.3f}, {stats['q97_5']:.3f}]")
    
    # 10. Posterior Predictive Checks
    print("\n10. Running posterior predictive checks...")
    
    with model:
        ppc = bo.run_posterior_predictive(model, idata, kind="proportions")
    
    print("   ✓ Posterior predictive checks completed")
    
    # 11. Visualization
    print("\n11. Creating visualizations...")
    
    # Plot parameter distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cutpoints
    az.plot_posterior(idata, var_names=["alpha"], ax=axes[0, 0])
    axes[0, 0].set_title("Cutpoint Estimates")
    
    # Coefficients
    az.plot_posterior(idata, var_names=["beta"], ax=axes[0, 1])
    axes[0, 1].set_title("Coefficient Estimates")
    
    # Trace plots
    az.plot_trace(idata, var_names=["alpha"], ax=axes[1, 0])
    axes[1, 0].set_title("Cutpoint Trace")
    
    az.plot_trace(idata, var_names=["beta"], ax=axes[1, 1])
    axes[1, 1].set_title("Coefficient Trace")
    
    plt.tight_layout()
    plt.show()
    
    # 12. Model Comparison (if multiple models)
    print("\n12. Model comparison...")
    
    # Build alternative model (probit)
    model_probit = bo.cumulative_model(
        y=y, X=X_scaled, K=N_RESPONSE_CLASSES,
        priors=priors, link="probit",
        model_name="ordinal_probit"
    )
    
    with model_probit:
        idata_probit = pm.sample(
            draws=1000, tune=500, chains=4,
            return_inferencedata=True, random_seed=42
        )
    
    # Compare models
    comparison = bo.compare_models(
        {"logit": model, "probit": model_probit},
        {"logit": idata, "probit": idata_probit},
        ic="loo"
    )
    
    print("   Model comparison (LOO-CV):")
    print(comparison)
    
    # 13. Summary
    print("\n=== Summary ===")
    print(f"Best model: {comparison.index[0]}")
    print(f"LOO difference: {comparison.iloc[0]['elpd_diff']:.3f}")
    print(f"Model converged: {model_summary['diagnostics']['converged']}")
    print(f"Analysis completed successfully!")
    
    return {
        "model": model,
        "idata": idata,
        "model_summary": model_summary,
        "comparison": comparison,
        "validation_results": validation_results
    }

if __name__ == "__main__":
    results = main() 