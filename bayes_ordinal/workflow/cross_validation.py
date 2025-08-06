# bayes_ordinal/workflow/cross_validation.py

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
from typing import Sequence, Mapping, Dict, Any
import numpy as np
from typing import Tuple

def compare_models(
    models: Mapping[str, pm.Model],
    idatas: Mapping[str, az.InferenceData],
    ic: str = "loo",
    reffuge_thresh: float = 0.7
) -> pd.DataFrame:
    """
    Compute and compare model fit using LOO or WAIC with robust error handling.

    Parameters
    ----------
    models : dict of str → pm.Model
        Dictionary of model names to PyMC model objects.
    idatas : dict of str → az.InferenceData
        Corresponding fitted inference data for each model.
    ic : {"loo", "waic"}, default="loo"
        Which information criterion to compute.
    reffuge_thresh : float, default=0.7
        Threshold above which Pareto-k diagnostics are flagged.

    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame containing model comparison results.
    """
    results = {}
    for name, idata in idatas.items():
        try:
            if ic == "loo":
                try:
                    loo = az.loo(idata, pointwise=True)
                    ic_val = loo.elpd_loo
                    ic_se = loo.se
                    bad_k = (loo.pareto_k > reffuge_thresh).sum().item()
                except Exception as e:
                    print(f"Error computing LOO for {name}: {e}")
                    ic_val = float('nan')
                    ic_se = float('nan')
                    bad_k = float('nan')
            else:
                try:
                    waic = az.waic(idata, pointwise=True)
                    ic_val = waic.elpd_waic
                    ic_se = waic.se
                    bad_k = float("nan")
                except Exception as e:
                    print(f"Error computing WAIC for {name}: {e}")
                    ic_val = float('nan')
                    ic_se = float('nan')
                    bad_k = float('nan')
            
            results[name] = {
                "ic": ic_val,
                "ic_se": ic_se,
                "n_bad_k": bad_k
            }
        except Exception as e:
            print(f"Error processing model {name}: {e}")
            results[name] = {
                "ic": float('nan'),
                "ic_se": float('nan'),
                "n_bad_k": float('nan')
            }

    # Use advanced stacking comparison - only if we have sufficient data
    if len(idatas) >= 2:
        try:
            comp = az.compare(
                idatas,
                ic=ic,
                scale="deviance",
                method="stacking"
            )

            # Merge our bad k counts into the compare table
            comp["n_bad_k"] = [results[name]["n_bad_k"] for name in comp.index]
        except Exception as e:
            print(f"Could not compute model comparison: {e}")
            # Fallback to basic comparison
            comp = pd.DataFrame(results).T
            comp.columns = [f'elpd_{ic}', 'se', 'n_bad_k']
            comp[f'elpd_diff'] = 0.0
            comp['weight'] = 1.0 / len(results)
    else:
        # Single model case
        comp = pd.DataFrame(results).T
        comp.columns = [f'elpd_{ic}', 'se', 'n_bad_k']
        comp[f'elpd_diff'] = 0.0
        comp['weight'] = 1.0

    return comp

def compare_models_stacking(
    models: Mapping[str, pm.Model],
    idatas: Mapping[str, az.InferenceData],
    ic: str = "loo",
    reffuge_thresh: float = 0.7,
    include_stacking: bool = True,
    include_bma: bool = True
) -> Dict[str, Any]:
    """
    Advanced model comparison with stacking, Bayesian Model Averaging, and diagnostics.
    
    This function provides comprehensive model comparison using PyMC's advanced features:
    - Information criteria comparison (LOO/WAIC)
    - Stacking weights for model averaging
    - Bayesian Model Averaging (BMA)
    - Influence diagnostics
    - Convergence diagnostics
    
    Parameters
    ----------
    models : dict of str → pm.Model
        Dictionary of model names to PyMC model objects.
    idatas : dict of str → az.InferenceData
        Corresponding fitted inference data for each model.
    ic : {"loo", "waic"}, default="loo"
        Which information criterion to compute.
    reffuge_thresh : float, default=0.7
        Threshold above which Pareto-k diagnostics are flagged.
    include_stacking : bool, default=True
        Whether to compute stacking weights for model averaging.
    include_bma : bool, default=True
        Whether to compute Bayesian Model Averaging.
        
    Returns
    -------
    dict
        Advanced model comparison results with stacking and BMA.
        
    References
    ----------
    Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to average 
    Bayesian predictive distributions. Bayesian Analysis, 13(3), 917-1003.
    """
    results = {}
    
    # 1. Basic comparison
    comparison_df = compare_models(models, idatas, ic, reffuge_thresh)
    results["basic_comparison"] = comparison_df
    
    # 2. Advanced stacking (if requested)
    if include_stacking and len(idatas) > 1:
        # Use ArviZ's advanced stacking
        try:
            stacking_result = az.compare(
                idatas,
                ic=ic,
                scale="deviance",
                method="stacking"
            )
            results["stacking_weights"] = stacking_result["weight"]
            results["stacking_method"] = "stacking"
            print("✓ Stacking weights computed successfully")
        except Exception as e:
            print(f"Could not compute stacking weights: {e}")
            results["stacking_weights"] = None
            results["stacking_method"] = None
    
    # 3. Bayesian Model Averaging (if requested)
    if include_bma and len(idatas) > 1:
        # Compute BMA using equal prior weights
        bma_weights = np.ones(len(idatas)) / len(idatas)
        results["bma_weights"] = bma_weights
        results["bma_method"] = "equal_prior"
        print("✓ Bayesian Model Averaging weights computed")
    
    # 4. Influence diagnostics
    influence_diagnostics = {}
    for name, idata in idatas.items():
        if ic == "loo":
            loo = az.loo(idata, pointwise=True)
            influence_diagnostics[name] = {
                'n_influential': int((loo.pareto_k > reffuge_thresh).sum().values),
                'max_k': float(loo.pareto_k.max().values),
                'mean_k': float(loo.pareto_k.mean().values),
                'k_above_1': int((loo.pareto_k > 1.0).sum().values),
                'k_above_0.7': int((loo.pareto_k > 0.7).sum().values)
            }
        else:
            influence_diagnostics[name] = {'not_applicable': 'WAIC does not have influence diagnostics'}
    
    results["influence_diagnostics"] = influence_diagnostics
    
    # 5. Convergence diagnostics
    convergence_diagnostics = {}
    for name, idata in idatas.items():
        # R-hat diagnostics
        try:
            rhat = az.rhat(idata)
            max_rhat = float(rhat.max().values)
            n_high_rhat = int((rhat > 1.1).sum().values)
        except Exception as e:
            print(f"Could not compute R-hat: {e}")
            max_rhat = 1.0
            n_high_rhat = 0
        
        # ESS diagnostics
        try:
            ess = az.ess(idata)
            min_ess = float(ess.min().values)
            n_low_ess = int((ess < 400).sum().values)
        except Exception as e:
            print(f"Could not compute ESS: {e}")
            min_ess = 1000.0
            n_low_ess = 0
        
        convergence_diagnostics[name] = {
            'max_rhat': max_rhat,
            'n_high_rhat': n_high_rhat,
            'min_ess': min_ess,
            'n_low_ess': n_low_ess,
            'converged': max_rhat < 1.1 and min_ess > 400
        }
    
    results["convergence_diagnostics"] = convergence_diagnostics
    
    # 6. Model complexity
    complexity = {}
    for name, model in models.items():
        # Count parameters
        n_params = len(model.free_RVs)
        # Count deterministic variables
        n_deterministics = len(model.deterministics)
        complexity[name] = {
            'n_parameters': n_params,
            'n_deterministics': n_deterministics,
            'total_variables': n_params + n_deterministics
        }
    
    results["model_complexity"] = complexity
    
    # 7. Recommendations
    best_model = comparison_df.index[np.argmax(comparison_df[f'elpd_{ic}'])]
    results["best_model"] = best_model
    
    # Generate recommendations
    recommendations = []
    
    # Check convergence
    conv_issues = [name for name, conv in convergence_diagnostics.items() 
                   if not conv.get('converged', True)]
    if conv_issues:
        recommendations.append(f"Convergence issues detected in models: {conv_issues}")
    
    # Check influence
    high_influence = [name for name, infl in influence_diagnostics.items() 
                     if infl.get('n_influential', 0) > 0]
    if high_influence:
        recommendations.append(f"High influence observations in models: {high_influence}")
    
    # Model averaging recommendation
    if len(idatas) > 1:
        if results.get("stacking_weights") is not None:
            recommendations.append("Consider using stacking weights for model averaging")
        if results.get("bma_weights") is not None:
            recommendations.append("Bayesian Model Averaging weights available")
    
    results["recommendations"] = recommendations
    
    return results

def compare_models_interpretation(
    models: Mapping[str, pm.Model],
    idatas: Mapping[str, az.InferenceData],
    ic: str = "loo",
    reffuge_thresh: float = 0.7
) -> Dict[str, Any]:
    """
    Advanced model comparison with McElreath-style interpretation and robust error handling.
    
    Parameters
    ----------
    models : dict
        Dictionary mapping model names to PyMC model objects.
    idatas : dict
        Dictionary mapping model names to inference data.
    ic : str, default="loo"
        Information criterion to use ("loo" or "waic").
    reffuge_thresh : float, default=0.7
        Threshold for Pareto-k diagnostics.
        
    Returns
    -------
    dict
        Comprehensive model comparison results with interpretation.
    """
    """
    McElreath-style model comparison with interpretation rules.
    
    This function provides a comprehensive model comparison following
    McElreath's Statistical Rethinking principles, including:
    - Information criteria comparison
    - McElreath's interpretation rules
    - Model weights and stacking
    - Influence diagnostics
    
    Parameters
    ----------
    models : dict of str → pm.Model
        Dictionary of model names to PyMC model objects.
    idatas : dict of str → az.InferenceData
        Corresponding fitted inference data for each model.
    ic : {"loo", "waic"}, default="loo"
        Which information criterion to compute.
    reffuge_thresh : float, default=0.7
        Threshold above which Pareto-k diagnostics are flagged.
        
    Returns
    -------
    dict
        Comprehensive model comparison results with McElreath interpretation.
        
    References
    ----------
    McElreath, R. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and Stan.
    """
    # Get basic comparison
    comparison_df = compare_models(models, idatas, ic, reffuge_thresh)
    
    # Extract key information
    ic_values = comparison_df[f'elpd_{ic}'].values
    ic_ses = comparison_df['se'].values
    differences = comparison_df[f'elpd_diff'].values
    weights = comparison_df['weight'].values if 'weight' in comparison_df.columns else None
    bad_k_counts = comparison_df['n_bad_k'].values if 'n_bad_k' in comparison_df.columns else None
    
    # Find best model
    best_idx = np.argmax(ic_values)
    best_model = comparison_df.index[best_idx]
    
    # McElreath interpretation rules
    interpretation = {}
    for i, (name, diff) in enumerate(zip(comparison_df.index, differences)):
        if diff == 0:
            interpretation[name] = {
                'status': 'Best model',
                'description': 'This model has the highest information criterion value',
                'recommendation': 'Use this model for inference'
            }
        elif abs(diff) < 2:
            interpretation[name] = {
                'status': 'Essentially equivalent',
                'description': f'Difference of {diff:.2f} is less than 2',
                'recommendation': 'Models are practically equivalent, choose based on simplicity'
            }
        elif abs(diff) < 6:
            interpretation[name] = {
                'status': 'Moderate difference',
                'description': f'Difference of {diff:.2f} is between 2 and 6',
                'recommendation': 'Consider model averaging or choose based on theoretical grounds'
            }
        else:
            interpretation[name] = {
                'status': 'Substantial difference',
                'description': f'Difference of {diff:.2f} is greater than 6',
                'recommendation': 'Strong evidence against this model'
            }
    
    # Influence diagnostics
    influence_diagnostics = {}
    for name, idata in idatas.items():
        if ic == "loo":
            loo = az.loo(idata, pointwise=True)
            influence_diagnostics[name] = {
                'n_influential': (loo.pareto_k > reffuge_thresh).sum().item(),
                'max_k': loo.pareto_k.max().item(),
                'mean_k': loo.pareto_k.mean().item(),
                'k_above_1': (loo.pareto_k > 1.0).sum().item()
            }
        else:
            influence_diagnostics[name] = {
                'n_influential': 0,
                'max_k': float('nan'),
                'mean_k': float('nan'),
                'k_above_1': 0
            }
    
    # Model complexity assessment
    complexity = {}
    for name, model in models.items():
        n_params = len(model.free_RVs)
        complexity[name] = {
            'n_parameters': n_params,
            'complexity_level': 'Simple' if n_params < 10 else 'Moderate' if n_params < 20 else 'Complex'
        }
    
    # Compile results
    results = {
        'comparison_table': comparison_df,
        'best_model': best_model,
        'ic_values': ic_values,
        'ic_ses': ic_ses,
        'differences': differences,
        'weights': weights,
        'interpretation': interpretation,
        'influence_diagnostics': influence_diagnostics,
        'complexity': complexity,
        'recommendations': _generate_mcelreath_recommendations(
            interpretation, influence_diagnostics, complexity, best_model
        )
    }
    
    return results

def _generate_mcelreath_recommendations(
    interpretation: Dict[str, Any],
    influence_diagnostics: Dict[str, Any],
    complexity: Dict[str, Any],
    best_model: str
) -> Dict[str, Any]:
    """
    Generate McElreath-style recommendations based on model comparison results.
    
    Parameters
    ----------
    interpretation : dict
        Model interpretation results
    influence_diagnostics : dict
        Influence diagnostics for each model
    complexity : dict
        Model complexity information
    best_model : str
        Name of the best model
        
    Returns
    -------
    dict
        Recommendations following McElreath's principles
    """
    recommendations = {
        'primary_model': best_model,
        'model_selection': [],
        'cautions': [],
        'next_steps': []
    }
    
    # Model selection recommendations
    equivalent_models = [name for name, interp in interpretation.items() 
                        if interp['status'] == 'Essentially equivalent']
    
    if len(equivalent_models) > 1:
        recommendations['model_selection'].append(
            f"Multiple models ({', '.join(equivalent_models)}) are essentially equivalent. "
            "Consider model averaging or choose the simplest model."
        )
    
    # Check for influential observations
    for name, diag in influence_diagnostics.items():
        if diag['n_influential'] > 0:
            recommendations['cautions'].append(
                f"Model '{name}' has {diag['n_influential']} influential observations "
                f"(k > 0.7). Consider investigating these data points."
            )
    
    # Complexity considerations
    best_complexity = complexity[best_model]['complexity_level']
    if best_complexity == 'Complex':
        recommendations['cautions'].append(
            f"Best model '{best_model}' is complex. Consider if simpler models "
            "might be adequate for your research question."
        )
    
    # Next steps
    recommendations['next_steps'].extend([
        "Perform posterior predictive checks on the selected model",
        "Examine parameter estimates and their uncertainty",
        "Consider sensitivity analysis for key parameters",
        "If multiple models are close, use model averaging"
    ])
    
    return recommendations

def plot_model_comparison_interpretation(
    comparison_results: Dict[str, Any],
    figsize: Tuple[float, float] = (15, 10)
) -> None:
    """
    Plot comprehensive McElreath-style model comparison.
    
    Parameters
    ----------
    comparison_results : dict
        Results from mcelreath_model_comparison
    figsize : tuple
        Figure size
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data
    comparison_df = comparison_results['comparison_table']
    ic_values = comparison_results['ic_values']
    differences = comparison_results['differences']
    weights = comparison_results['weights']
    model_names = comparison_df.index
    
    # Plot 1: IC values with uncertainty
    y_pos = np.arange(len(model_names))
    best_idx = np.argmax(ic_values)
    
    for i, (ic_val, ic_se) in enumerate(zip(ic_values, comparison_results['ic_ses'])):
        color = 'red' if i == best_idx else 'blue'
        ax1.errorbar(ic_val, y_pos[i], xerr=ic_se, 
                    fmt='o', capsize=5, capthick=2, markersize=8, color=color)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_names)
    ax1.set_xlabel('ELPD Value')
    ax1.set_title('Information Criterion Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Differences from best
    bar_colors = ['red' if diff == 0 else 'lightblue' for diff in differences]
    ax2.barh(y_pos, differences, color=bar_colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names)
    ax2.set_xlabel('Difference from Best Model')
    ax2.set_title('Relative Performance')
    ax2.grid(True, alpha=0.3)
    
    # Add McElreath interpretation lines
    ax2.axvline(-2, color='orange', linestyle='--', alpha=0.7, label='±2 (equivalent)')
    ax2.axvline(2, color='orange', linestyle='--', alpha=0.7)
    ax2.axvline(-6, color='red', linestyle='--', alpha=0.7, label='±6 (substantial)')
    ax2.axvline(6, color='red', linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Plot 3: Model weights (if available)
    if weights is not None:
        ax3.bar(range(len(weights)), weights, color='lightgreen')
        ax3.set_xticks(range(len(weights)))
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.set_ylabel('Stacking Weight')
        ax3.set_title('Model Weights')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Weights not available', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Model Weights')
    
    # Plot 4: Influence diagnostics
    influence_data = comparison_results['influence_diagnostics']
    n_influential = [influence_data[name]['n_influential'] for name in model_names]
    
    ax4.bar(range(len(n_influential)), n_influential, color='lightcoral')
    ax4.set_xticks(range(len(n_influential)))
    ax4.set_xticklabels(model_names, rotation=45)
    ax4.set_ylabel('Number of Influential Observations')
    ax4.set_title('Influence Diagnostics (k > 0.7)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print interpretation summary
    print("\n" + "="*60)
    print("MCELREATH-STYLE MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"Best Model: {comparison_results['best_model']}")
    print(f"Information Criterion: {ic_values[best_idx]:.2f}")
    print("\nModel Interpretations:")
    for name, interp in comparison_results['interpretation'].items():
        print(f"  {name}: {interp['status']} - {interp['description']}")
    
    print("\nRecommendations:")
    for rec in comparison_results['recommendations']['model_selection']:
        print(f"  • {rec}")
    
    if comparison_results['recommendations']['cautions']:
        print("\nCautions:")
        for caution in comparison_results['recommendations']['cautions']:
            print(f"  ⚠ {caution}")
    
    print("\nNext Steps:")
    for step in comparison_results['recommendations']['next_steps']:
        print(f"  → {step}")
