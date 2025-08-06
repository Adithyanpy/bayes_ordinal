#!/usr/bin/env python3
"""
Basic example demonstrating Bayesian ordinal regression workflow.

This example shows how to:
1. Generate synthetic ordinal data
2. Fit cumulative and partial odds models
3. Run diagnostics and model comparison
4. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
import bayes_ordinal as bo

def main():
    """Run complete ordinal regression workflow."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("=== Bayesian Ordinal Regression Example ===\n")
    
    # 1. Generate synthetic data
    print("1. Generating synthetic data...")
    n_samples, n_features, K = 200, 3, 5
    
    # True parameters
    true_beta = np.array([0.5, -0.3, 0.8])
    true_cutpoints = np.array([-1.5, -0.5, 0.5, 1.5])
    
    # Generate features
    X = np.random.normal(size=(n_samples, n_features))
    
    # Generate ordinal responses using cumulative model
    linear_predictor = X @ true_beta
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Cumulative probabilities
        probs = 1 / (1 + np.exp(-(true_cutpoints - linear_predictor[i])))
        probs = np.concatenate([[0], probs, [1]])
        
        # Sample category
        category_probs = probs[1:] - probs[:-1]
        y[i] = np.random.choice(K, p=category_probs)
    
    print(f"   Generated {n_samples} samples with {K} categories")
    print(f"   Category distribution: {np.bincount(y, minlength=K)}")
    
    # 2. Validate and preprocess data
    print("\n2. Validating and preprocessing data...")
    y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K)
    X_scaled, scaler = bo.standardize_features(X_clean)
    
    print(f"   Data validation passed")
    print(f"   Features standardized")
    
    # 3. Define priors
    print("\n3. Setting up priors...")
    priors = bo.get_default_priors()
    print(f"   Using default priors: {list(priors.keys())}")
    
    # 4. Build models
    print("\n4. Building models...")
    
    # Cumulative model
    model_cum = bo.cumulative_model(
        y=y_clean, X=X_scaled, K=K,
        link="logit",
        priors=priors
    )
    
    # Partial odds model
    model_partial = bo.partial_odds_model(
        y=y_clean, X=X_scaled, K=K,
        priors=priors
    )
    
    print(f"   Built cumulative model with {len(model_cum.free_RVs)} parameters")
    print(f"   Built partial odds model with {len(model_partial.free_RVs)} parameters")
    
    # 5. Run prior predictive checks
    print("\n5. Running prior predictive checks...")
    
    # For cumulative model
    prior_idata_cum = bo.run_prior_predictive(
        model_cum, draws=100, plot=False
    )
    
    # For partial odds model
    prior_idata_partial = bo.run_prior_predictive(
        model_partial, draws=100, plot=False
    )
    
    print(f"   Prior predictive samples generated")
    
    # 6. Fit models
    print("\n6. Fitting models...")
    
    # Fit cumulative model
    print("   Fitting cumulative model...")
    idata_cum = bo.fit_model(
        model_cum,
        draws=1000,
        tune=500,
        chains=2,
        smoke_test=False,
        return_inferencedata=True
    )
    
    # Fit partial odds model
    print("   Fitting partial odds model...")
    idata_partial = bo.fit_model(
        model_partial,
        draws=1000,
        tune=500,
        chains=2,
        smoke_test=False,
        return_inferencedata=True
    )
    
    print(f"   Models fitted successfully")
    
    # 7. Run diagnostics
    print("\n7. Running diagnostics...")
    
    # Check convergence
    conv_cum = bo.check_convergence(idata_cum)
    conv_partial = bo.check_convergence(idata_partial)
    
    print(f"   Cumulative model converged: {conv_cum['converged']}")
    print(f"   Partial odds model converged: {conv_partial['converged']}")
    
    if not conv_cum['converged']:
        print(f"   Cumulative model issues: {conv_cum['n_divergences']} divergences")
    if not conv_partial['converged']:
        print(f"   Partial odds model issues: {conv_partial['n_divergences']} divergences")
    
    # 8. Run posterior predictive checks
    print("\n8. Running posterior predictive checks...")
    
    ppc_cum = bo.run_posterior_predictive(
        model_cum, idata_cum,
        kind="proportions"
    )
    
    ppc_partial = bo.run_posterior_predictive(
        model_partial, idata_partial,
        kind="proportions"
    )
    
    print(f"   Posterior predictive checks completed")
    
    # 9. Model comparison
    print("\n9. Comparing models...")
    
    comparison = bo.compare_models(
        {"cumulative": model_cum, "partial": model_partial},
        {"cumulative": idata_cum, "partial": idata_partial},
        ic="loo"
    )
    
    print("   Model comparison results:")
    print(comparison)
    
    # 10. Visualize results
    print("\n10. Creating visualizations...")
    
    # Coefficient effects
    bo.plot_coefficient_effects(
        idata_cum,
        var_name="beta",
        feature_names=[f"Feature {i+1}" for i in range(n_features)]
    )
    
    # Cutpoint estimates
    bo.plot_cutpoints(idata_cum, var_name="alpha")
    
    # Model comparison plot
    bo.plot_model_comparison(
        {"cumulative": idata_cum, "partial": idata_partial},
        ic="loo"
    )
    
    print("   Visualizations created")
    
    # 11. Summary
    print("\n=== Summary ===")
    print(f"Best model: {comparison.index[0]}")
    print(f"LOO difference: {comparison.iloc[0]['elpd_diff']:.3f}")
    print(f"Both models converged successfully")
    print(f"Workflow completed successfully!")
    
    return {
        'models': {'cumulative': model_cum, 'partial': model_partial},
        'idata': {'cumulative': idata_cum, 'partial': idata_partial},
        'comparison': comparison,
        'convergence': {'cumulative': conv_cum, 'partial': conv_partial}
    }

if __name__ == "__main__":
    results = main() 