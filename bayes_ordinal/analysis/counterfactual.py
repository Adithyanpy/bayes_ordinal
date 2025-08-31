"""
Counterfactual analysis for Bayesian ordinal models.

This module provides functions for running counterfactual analysis on fitted
ordinal regression models.
"""

import numpy as np
from typing import Dict, Any, Tuple
import pymc as pm
import arviz as az
from scipy.stats import norm


def run_counterfactual_analysis(
    model: pm.Model,
    idata: Any,
    scenarios: Dict[str, Dict[str, int]],
    feature_names: list,
    n_samples: int = 1000
) -> Dict[str, Any]:
    """    
    This function implements the counterfactual analysis using pm.set_data() with
    xr.DataArray and pm.sample_posterior_predictive() to generate predictions.
    
    Parameters
    ----------
    model : pm.Model
        The fitted ordinal model
    idata : az.InferenceData
        Posterior samples from the model
    scenarios : Dict[str, Dict[str, int]]
        Dictionary mapping scenario names to feature values.
        Example: {"high_contact": {"contact": 1, "action": 1, "intention": 0}}
    feature_names : list
        Names of features in the model
    n_samples : int, default=1000
        Number of posterior predictive samples per scenario
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with counterfactual results including:
        - "scenarios": The input scenarios
        - "predictions": Posterior predictive samples for each scenario
        - "probabilities": Category probabilities for each scenario
        - "summary": Summary statistics
        
    Examples
    --------
    >>> scenarios = {
    ...     "baseline": {"action": 0, "intention": 0, "contact": 0},
    ...     "high_risk": {"action": 1, "intention": 1, "contact": 1}
    ... }
    >>> results = run_counterfactual_analysis(model, idata, scenarios, feature_names)
    """
    results = {
        "scenarios": scenarios,
        "predictions": {},
        "probabilities": {},
        "summary": {}
    }
    
    # Check if model has pm.Data containers for features
    has_data_containers = False
    data_vars = {}
    
    # Look for pm.Data containers in the model
    for name in feature_names:
        var_name = name.lower().replace(" ", "_")
        if var_name in model.named_vars and hasattr(model[var_name], 'get_value'):
            has_data_containers = True
            data_vars[var_name] = model[var_name]
    
    if not has_data_containers:
        # Model doesn't have pm.Data containers - use alternative approach
        print(" Model doesn't have pm.Data containers. Using alternative counterfactual approach...")
        print(" This approach works with your generic beta coefficient structure")
        print(" It computes predictions using the posterior samples directly")
        return _run_counterfactual_without_data_containers(
            model, idata, scenarios, feature_names, n_samples
        )
    
    # Original data containers (to restore later)
    original_data = {}
    
    with model:
        # Store original data values
        for name in feature_names:
            var_name = name.lower().replace(" ", "_")
            if var_name in model.named_vars:
                original_data[var_name] = model[var_name].get_value()
        
        # Run counterfactuals for each scenario
        for scenario_name, scenario_values in scenarios.items():
            print(f"Running counterfactual for scenario: {scenario_name}")
            
            scenario_data = {}
            for name in feature_names:
                var_name = name.lower().replace(" ", "_")
                value = scenario_values.get(name, 0)  # Default to 0 if not specified
                scenario_data[var_name] = np.array([value], dtype=float)
            
            # Update model data using pm.set_data
            pm.set_data(scenario_data, model=model)
            
            # Sample posterior predictive for this scenario
            ppc = pm.sample_posterior_predictive(
                idata,
                predictions=True,
                extend_inferencedata=False,
                random_seed=42
            )
            
            # Store predictions  
            pred_var = None
            for var in ppc.predictions.data_vars:
                if "y" in var or "response" in var:
                    pred_var = var
                    break
            
            if pred_var:
                predictions = ppc.predictions[pred_var].values.flatten()
                results["predictions"][scenario_name] = predictions
                
                # Calculate category probabilities
                unique_vals, counts = np.unique(predictions, return_counts=True)
                probs = counts / len(predictions)
                prob_dict = {int(val): float(prob) for val, prob in zip(unique_vals, probs)}
                results["probabilities"][scenario_name] = prob_dict
                
                # Summary statistics
                results["summary"][scenario_name] = {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "median": float(np.median(predictions)),
                    "mode": int(unique_vals[np.argmax(counts)])
                }
            else:
                print(f"Warning: Could not find prediction variable for scenario {scenario_name}")
        
        # Restore original data
        if original_data:
            pm.set_data(original_data, model=model)
    
    return results


def _run_counterfactual_without_data_containers(
    model: pm.Model,
    idata: Any,
    scenarios: Dict[str, Dict[str, int]],
    feature_names: list,
    n_samples: int = 1000
) -> Dict[str, Any]:
    """
    Alternative counterfactual analysis for models without pm.Data containers.
    
    This function creates new data matrices for each scenario and uses
    the model's linear predictor to compute predictions.
    """
    results = {
        "scenarios": scenarios,
        "predictions": {},
        "probabilities": {},
        "summary": {}
    }
    
    # Auto-detect beta and cutpoint variables
    available_vars = list(idata.posterior.data_vars.keys())
    print(f" Available variables in posterior: {available_vars}")
    
    # Find beta variable
    beta_var_name = None
    beta_patterns = ['beta', 'coefficients', 'coef', 'slopes']
    for pattern in beta_patterns:
        matching_vars = [v for v in available_vars if pattern in v.lower()]
        if matching_vars:
            beta_var_name = matching_vars[0]
            break
    
    # Find cutpoints variable
    cutpoint_var_name = None
    cutpoint_patterns = ['cutpoints', 'cuts', 'thresholds', 'alpha']
    for pattern in cutpoint_patterns:
        matching_vars = [v for v in available_vars if pattern in v.lower()]
        if matching_vars:
            cutpoint_var_name = matching_vars[0]
            break
    
    if beta_var_name is None or cutpoint_var_name is None:
        print(f" Could not find required variables:")
        print(f"   Beta variable found: {beta_var_name}")
        print(f"   Cutpoints variable found: {cutpoint_var_name}")
        print(f" Available variables: {available_vars}")
        return results
    
    print(f" Using variables: beta={beta_var_name}, cutpoints={cutpoint_var_name}")
    
    # Get posterior samples
    try:
        beta_samples = idata.posterior[beta_var_name].values  # Shape: (chains, draws, n_features)
        cutpoint_samples = idata.posterior[cutpoint_var_name].values  # Shape: (chains, draws, K-1)
    except KeyError as e:
        print(f" Error accessing posterior samples: {e}")
        return results
    
    # Run counterfactuals for each scenario
    for scenario_name, scenario_values in scenarios.items():
        print(f"Running counterfactual for scenario: {scenario_name}")
        
        # Create feature matrix for this scenario
        scenario_X = np.zeros((1, len(feature_names)))
        for i, name in enumerate(feature_names):
            scenario_X[0, i] = scenario_values.get(name, 0)
        
        # Compute linear predictor for all posterior samples
        # beta_samples shape: (chains, draws, n_features)
        # scenario_X shape: (1, n_features)
        # eta shape: (chains, draws, 1)
        eta = np.einsum('ijk,lk->ijl', beta_samples, scenario_X)
        
        # Detect link function from model
        is_probit = getattr(model, "link", "logit") == "probit"
        print(f"  Link function detected: {'probit' if is_probit else 'logit'}")
        
        # Define link function
        def F(z):
            return norm.cdf(z) if is_probit else 1.0 / (1.0 + np.exp(-z))
        
        # Vectorized computation for all posterior samples
        # Reshape samples for vectorized operations
        B = beta_samples.reshape(-1, beta_samples.shape[-1])   # (S, p) where S = chains * draws
        C = cutpoint_samples.reshape(-1, cutpoint_samples.shape[-1])  # (S, K-1)
        
        print(f"  Beta samples shape: {beta_samples.shape} -> {B.shape}")
        print(f"  Cutpoint samples shape: {cutpoint_samples.shape} -> {C.shape}")
        
        # Compute linear predictor for all samples at once
        eta = B @ scenario_X.ravel()  # (S,)
        print(f"  Linear predictor shape: {eta.shape}")
        
        # Compute cumulative probabilities using vectorized link function
        Fvals = F(C - eta[:, None])  # (S, K-1)
        print(f"  CDF values shape: {Fvals.shape}")
        
        # Convert to category probabilities: prepend 0, append 1, then diff
        cdf = np.concatenate([
            np.zeros((len(eta), 1)), 
            Fvals, 
            np.ones((len(eta), 1))
        ], axis=1)  # (S, K+1)
        
        probs = np.diff(cdf, axis=1)  # (S, K)
        print(f"  Category probabilities shape: {probs.shape}")
        
        # Sample categories from multinomial distribution for each row
        all_predictions = []
        for i in range(probs.shape[0]):
            # Ensure probabilities are valid (non-negative and sum to 1)
            prob_row = np.maximum(probs[i], 0)
            prob_row = prob_row / prob_row.sum()
            category = np.random.choice(probs.shape[1], p=prob_row)
            all_predictions.append(category)
        
        print(f"  Generated {len(all_predictions)} predictions for {scenario_name}")
        
        # Store results
        results["predictions"][scenario_name] = np.array(all_predictions)
        
        # Calculate category probabilities
        unique_vals, counts = np.unique(all_predictions, return_counts=True)
        probs = counts / len(all_predictions)
        prob_dict = {int(val): float(prob) for val, prob in zip(unique_vals, probs)}
        results["probabilities"][scenario_name] = prob_dict
        
        # Summary statistics
        results["summary"][scenario_name] = {
            "mean": float(np.mean(all_predictions)),
            "std": float(np.std(all_predictions)),
            "median": float(np.median(all_predictions)),
            "mode": int(unique_vals[np.argmax(counts)])
        }
        
        print(f"  Stored results for {scenario_name}: mean={results['summary'][scenario_name]['mean']:.3f}")
    
    print(f"Final results structure: {list(results.keys())}")
    print(f"Scenarios processed: {list(results['summary'].keys())}")
    return results


def plot_counterfactual_results(
    results: Dict[str, Any],
    figsize: Tuple[float, float] = (12, 8)
) -> None:
    """
    Plot counterfactual analysis results.
    
    This function creates histograms using plt.bar().
    
    Parameters
    ----------
    results : dict
        Results from run_counterfactual_analysis()
    figsize : tuple
        Figure size
        
    Examples
    --------
    >>> results = run_counterfactual_analysis(model, idata, scenarios, feature_names)
    >>> plot_counterfactual_results(results)
    """
    import matplotlib.pyplot as plt
    
    n_scenarios = len(results)
    
    # Handle different result formats
    if "predictions" in results:
        scenarios_data = results.get("predictions", {})
    else:
        scenarios_data = results
    
    if not scenarios_data:
        print("No scenario data found to plot")
        return
    
    # Create subplots
    n_cols = min(4, len(scenarios_data))
    n_rows = (len(scenarios_data) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each scenario
    for i, (scenario_name, predictions) in enumerate(scenarios_data.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Count occurrences of each category
            unique_vals, counts = np.unique(predictions, return_counts=True)
            
            # Create bar plot
            bars = ax.bar(unique_vals, counts, alpha=0.7, color='skyblue', edgecolor='navy')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}', ha='center', va='bottom')
            
            ax.set_xlabel('Response Category')
            ax.set_ylabel('Count')
            ax.set_title(f'Scenario: {scenario_name}')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to show all categories
            ax.set_xticks(unique_vals)
    
    # Hide unused subplots
    for i in range(len(scenarios_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
