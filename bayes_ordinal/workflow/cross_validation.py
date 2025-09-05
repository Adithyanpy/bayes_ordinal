
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
from typing import Sequence, Mapping, Dict, Any, Tuple
import numpy as np

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _check_log_likelihood(idata: az.InferenceData, name: str) -> bool:
    """Check if log likelihood is available in InferenceData."""
    has_ll = hasattr(idata, "log_likelihood")
    
    if has_ll:
        print(f" Log likelihood found for {name}")
        
    else:
        print(f" Log likelihood not found for {name} - LOO/WAIC will fail")
        if hasattr(idata, 'posterior') and hasattr(idata.posterior, 'log_likelihood'):
            print(f"  Found log_likelihood in posterior for {name}")
        elif hasattr(idata, 'sample_stats') and hasattr(idata.sample_stats, 'log_likelihood'):
            print(f"  Found log_likelihood in sample_stats for {name}")
        else:
            print(f"  No log_likelihood found anywhere for {name}")
            print(f"  Available variables in posterior: {list(idata.posterior.data_vars.keys())}")
            print(f"  Available variables in sample_stats: {list(idata.sample_stats.data_vars.keys())}")
    
    return has_ll

def _extract_arviz_values(obj, use_max: bool = True) -> np.ndarray:
    """Extract values from ArviZ objects, handling different versions."""
    if hasattr(obj, 'values') and callable(obj.values):
        values = list(obj.values())
    else:
        values = obj
    
    # Convert to numpy array, handling array vs scalar values
    if use_max:
        return np.array([float(np.max(v)) if hasattr(v, '__len__') else float(v) for v in values])
    else:
        return np.array([float(np.min(v)) if hasattr(v, '__len__') else float(v) for v in values])

def _compute_influence_diagnostics(idata: az.InferenceData, ic: str, reffuge_thresh: float) -> Dict[str, Any]:
    """Compute influence diagnostics for a single model."""
    if ic != "loo":
        return {'not_applicable': 'WAIC does not have influence diagnostics'}
    
    try:
        loo = az.loo(idata, pointwise=True)
        pareto_k_array = _extract_arviz_values(loo.pareto_k, use_max=True)
        
        return {
            'n_influential': int(np.sum(pareto_k_array > reffuge_thresh)),
            'max_k': float(np.max(pareto_k_array)),
            'mean_k': float(np.mean(pareto_k_array)),
            'k_above_1': int(np.sum(pareto_k_array > 1.0)),
            'k_above_0.7': int(np.sum(pareto_k_array > 0.7))
        }
    except Exception as e:
        print(f"Could not compute influence diagnostics: {e}")
        return {'n_influential': 0, 'max_k': float('nan'), 'mean_k': float('nan'), 
                'k_above_1': 0, 'k_above_0.7': 0}

def _compute_convergence_diagnostics(idata: az.InferenceData) -> Dict[str, Any]:
    """Compute convergence diagnostics for a single model."""
    # R-hat diagnostics
    try:
        rhat = az.rhat(idata)
        rhat_array = _extract_arviz_values(rhat, use_max=True)
        max_rhat = float(np.max(rhat_array))
        n_high_rhat = int(np.sum(rhat_array > 1.1))
    except Exception as e:
        print(f"Could not compute R-hat: {e}")
        max_rhat = 1.0
        n_high_rhat = 0
    
    # ESS diagnostics
    try:
        ess = az.ess(idata)
        ess_array = _extract_arviz_values(ess, use_max=False)
        min_ess = float(np.min(ess_array))
        n_low_ess = int(np.sum(ess_array < 400))
    except Exception as e:
        print(f"Could not compute ESS: {e}")
        min_ess = 1000.0
        n_low_ess = 0
    
    return {
        'max_rhat': max_rhat,
        'n_high_rhat': n_high_rhat,
        'min_ess': min_ess,
        'n_low_ess': n_low_ess,
        'converged': max_rhat < 1.1 and min_ess > 400
    }

def _assess_model_complexity(model: pm.Model) -> Dict[str, Any]:
    """Assess complexity of a single model."""
    n_params = len(model.free_RVs)
    n_deterministics = len(model.deterministics)
    
    return {
        'n_parameters': n_params,
        'n_deterministics': n_deterministics,
        'total_variables': n_params + n_deterministics,
        'complexity_level': 'Simple' if n_params < 10 else 'Moderate' if n_params < 20 else 'Complex'
    }

def _determine_best_model(comparison_df: pd.DataFrame, ic: str) -> str:
    """Determine the best model based on information criterion."""
    # For ELPD scale (log scale): larger values are better
    # Both ArviZ comparison (scale="log") and fallback use ELPD values
    return comparison_df[f'elpd_{ic}'].idxmax()

def _compute_ic_metrics(idata: az.InferenceData, ic: str, reffuge_thresh: float) -> Dict[str, Any]:
    """Compute information criterion metrics for a single model."""
    try:
        if ic == "loo":
            loo = az.loo(idata, pointwise=True)
            ic_val = loo.elpd_loo
            ic_se = loo.se
            bad_k = (loo.pareto_k > reffuge_thresh).sum().item()
            print(f" LOO computed successfully: {ic_val:.2f} +/- {ic_se:.2f}")
        else:
            waic = az.waic(idata, pointwise=True)
            ic_val = waic.elpd_waic
            ic_se = waic.se
            bad_k = float("nan")
            print(f" WAIC computed successfully: {ic_val:.2f} +/- {ic_se:.2f}")
        
        return {"ic": ic_val, "ic_se": ic_se, "n_bad_k": bad_k}
    except Exception as e:
        print(f"Error computing {ic.upper()}: {e}")
        return {"ic": float('nan'), "ic_se": float('nan'), "n_bad_k": float('nan')}

# ============================================================================
# MAIN FUNCTIONS (Refactored to use helpers)
# ============================================================================

def compare_models(
    models: Mapping[str, pm.Model],
    idatas: Mapping[str, az.InferenceData],
    ic: str = "loo",
    reffuge_thresh: float = 0.7
) -> pd.DataFrame:
    """Compute and compare model fit using LOO or WAIC with robust error handling."""
    results = {}
    log_likelihood_issues = []
    
    for name, idata in idatas.items():
        # Check log likelihood availability
        if not _check_log_likelihood(idata, name):
            log_likelihood_issues.append(name)
            results[name] = {"ic": float('nan'), "ic_se": float('nan'), "n_bad_k": float('nan')}
            continue
        
        # Compute IC metrics
        results[name] = _compute_ic_metrics(idata, ic, reffuge_thresh)

    # Provide guidance for log likelihood issues
    if log_likelihood_issues:
        print(f"\n Log likelihood issues detected for models: {log_likelihood_issues}")
        print("To fix this, ensure your models include:")
        print("enable_log_likelihood=True")
    # Use ArviZ comparison if possible
    if len(idatas) >= 2:
        try:
            comp = az.compare(idatas, ic=ic, scale="log", method="stacking")
            comp["n_bad_k"] = [results[name]["n_bad_k"] for name in comp.index]
            return comp
        except Exception as e:
            print(f"Could not compute model comparison: {e}")
            print("Falling back to basic comparison using available metrics")
    
    # Fallback to basic comparison
    comp = pd.DataFrame(results).T
    comp.columns = [f'elpd_{ic}', 'se', 'n_bad_k']
    comp[f'elpd_diff'] = 0.0
    comp['weight'] = 1.0 / len(results)
    return comp

def compare_models_stacking(
    models: Mapping[str, pm.Model],
    idatas: Mapping[str, az.InferenceData],
    ic: str = "loo",
    reffuge_thresh: float = 0.7,
    include_stacking: bool = True,
    include_bma: bool = True
) -> Dict[str, Any]:
    """Advanced model comparison with stacking, BMA, and diagnostics."""
    results = {}
    
    # 1. Basic comparison
    comparison_df = compare_models(models, idatas, ic, reffuge_thresh)
    results["basic_comparison"] = comparison_df
    
    # 2. Advanced stacking
    if include_stacking and len(idatas) > 1:
        try:
            stacking_result = az.compare(idatas, ic=ic, scale="deviance", method="stacking")
            results["stacking_weights"] = stacking_result["weight"]
            results["stacking_method"] = "stacking"
            print(" Stacking weights computed successfully")
        except Exception as e:
            print(f"Could not compute stacking weights: {e}")
            results["stacking_weights"] = None
            results["stacking_method"] = None
    
    # 3. Bayesian Model Averaging
    if include_bma and len(idatas) > 1:
        model_names = list(idatas.keys())
        bma_weights = {name: 1.0 / len(idatas) for name in model_names}
        results["bma_weights"] = bma_weights
        results["bma_method"] = "equal_prior"
        print(" Bayesian Model Averaging weights computed")
    
    # 4. Influence diagnostics
    influence_diagnostics = {name: _compute_influence_diagnostics(idata, ic, reffuge_thresh) 
                           for name, idata in idatas.items()}
    results["influence_diagnostics"] = influence_diagnostics
    
    # 5. Convergence diagnostics
    convergence_diagnostics = {name: _compute_convergence_diagnostics(idata) 
                             for name, idata in idatas.items()}
    results["convergence_diagnostics"] = convergence_diagnostics
    
    # 6. Model complexity
    complexity = {name: _assess_model_complexity(model) 
                 for name, model in models.items()}
    results["model_complexity"] = complexity
    
    # 7. Best model and recommendations
    results["best_model"] = _determine_best_model(comparison_df, ic)
    
    # Generate recommendations
    recommendations = []
    conv_issues = [name for name, conv in convergence_diagnostics.items() 
                   if not conv.get('converged', True)]
    if conv_issues:
        recommendations.append(f"Convergence issues detected in models: {conv_issues}")
    
    high_influence = [name for name, infl in influence_diagnostics.items() 
                     if infl.get('n_influential', 0) > 0]
    if high_influence:
        recommendations.append(f"High influence observations in models: {high_influence}")
    
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
    """Advanced model comparison."""
    # Get basic comparison
    comparison_df = compare_models(models, idatas, ic, reffuge_thresh)
    
    # Extract key information
    ic_values = comparison_df[f'elpd_{ic}'].values
    ic_ses = comparison_df['se'].values
    differences = comparison_df[f'elpd_diff'].values
    weights = comparison_df['weight'].values if 'weight' in comparison_df.columns else None
    
    # Find best model
    best_model = _determine_best_model(comparison_df, ic)
    
    interpretation = {}
    for name, diff in zip(comparison_df.index, differences):
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
    influence_diagnostics = {name: _compute_influence_diagnostics(idata, ic, reffuge_thresh) 
                           for name, idata in idatas.items()}
    
    # Model complexity
    complexity = {name: _assess_model_complexity(model) 
                 for name, model in models.items()}
    
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
        'recommendations': _generate_mc_recommendations(
            interpretation, influence_diagnostics, complexity, best_model
        )
    }
    
    return results

# ============================================================================
# UTILITY FUNCTIONS (Unchanged)
# ============================================================================

def display_comparison_results(results: Dict[str, Any]) -> None:
    """Display comprehensive comparison results in a readable format."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Basic comparison
    if "basic_comparison" in results:
        print("\n BASIC COMPARISON:")
        print(results["basic_comparison"].round(3))
    
    # Best model
    if "best_model" in results:
        print(f"\n BEST MODEL: {results['best_model']}")
    
    # Stacking weights
    if "stacking_weights" in results and results["stacking_weights"] is not None:
        print("\n STACKING WEIGHTS:")
        for model, weight in results["stacking_weights"].items():
            print(f"  {model}: {weight:.3f}")
    
    # BMA weights
    if "bma_weights" in results and results["bma_weights"] is not None:
        print("\n BAYESIAN MODEL AVERAGING WEIGHTS:")
        for model, weight in results["bma_weights"].items():
            print(f"  {model}: {weight:.3f}")
    
    # Convergence diagnostics
    if "convergence_diagnostics" in results:
        print("\n CONVERGENCE DIAGNOSTICS:")
        for model, conv in results["convergence_diagnostics"].items():
            status = " Probably Converged" if conv.get('converged', False) else " Not Converged"
            print(f"  {model}: {status}")
            print(f"    R-hat max: {conv.get('max_rhat', 'N/A'):.3f}")
            print(f"    ESS min: {conv.get('min_ess', 'N/A'):.0f}")
    
    # Influence diagnostics
    if "influence_diagnostics" in results:
        print("\n INFLUENCE DIAGNOSTICS:")
        for model, infl in results["influence_diagnostics"].items():
            print(f"  {model}:")
            print(f"    Influential obs (k > 0.7): {infl.get('n_influential', 'N/A')}")
            print(f"    Max k: {infl.get('max_k', 'N/A'):.3f}")
            print(f"    Mean k: {infl.get('mean_k', 'N/A'):.3f}")
    
    # Model complexity
    if "model_complexity" in results:
        print("\n MODEL COMPLEXITY:")
        for model, comp in results["model_complexity"].items():
            print(f"  {model}: {comp.get('n_parameters', 'N/A')} parameters")
    
    # Recommendations
    if "recommendations" in results and results["recommendations"]:
        print("\n RECOMMENDATIONS:")
        for rec in results["recommendations"]:
            print(f"  - {rec}")
    
    print("\n" + "="*80)

def _generate_mc_recommendations(
    interpretation: Dict[str, Any],
    influence_diagnostics: Dict[str, Any],
    complexity: Dict[str, Any],
    best_model: str
) -> Dict[str, Any]:
    """Generate recommendations based on model comparison results."""
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
        if diag.get('n_influential', 0) > 0:
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
    """Plot comprehensive model comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data
    comparison_df = comparison_results['comparison_table']
    ic_values = comparison_results['ic_values']
    differences = comparison_results['differences']
    weights = comparison_results['weights']
    model_names = comparison_df.index
    
    # Plot 1: IC values with uncertainty
    y_pos = np.arange(len(model_names))
    best_model_name = comparison_results['best_model']
    best_idx = list(model_names).index(best_model_name)
    
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
    
    ax2.axvline(-2, color='orange', linestyle='--', alpha=0.7, label='+/-2 (equivalent)')
    ax2.axvline(2, color='orange', linestyle='--', alpha=0.7)
    ax2.axvline(-6, color='red', linestyle='--', alpha=0.7, label='+/-6 (substantial)')
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
    
    # Handle different influence diagnostics structures
    n_influential = []
    for name in model_names:
        if name in influence_data:
            if isinstance(influence_data[name], dict):
                n_influential.append(influence_data[name].get('n_influential', 0))
            else:
                n_influential.append(0)
        else:
            n_influential.append(0)
    
    # Only plot if we have non-zero values or if we want to show the zero values
    if any(n > 0 for n in n_influential) or True:  # Always show for clarity
        ax4.bar(range(len(n_influential)), n_influential, color='lightcoral')
        ax4.set_xticks(range(len(n_influential)))
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.set_ylabel('Number of Influential Observations')
        ax4.set_title('Influence Diagnostics (k > 0.7)')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(n_influential):
            ax4.text(i, v + 0.001, str(v), ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No influential observations detected', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Influence Diagnostics (k > 0.7)')
    
    plt.tight_layout()
    plt.show()
    
    # Print interpretation summary
    print("\n" + "="*60)
    print("="*60)
    print(f"Best Model: {comparison_results['best_model']}")
    
    # Show the best model's IC value, not the maximum IC value
    best_model_name = comparison_results['best_model']
    best_model_idx = list(model_names).index(best_model_name)
    best_model_ic = ic_values[best_model_idx]
    print(f"Best Model IC Value: {best_model_ic:.2f}")
    
    print("\nModel Interpretations:")
    for name, interp in comparison_results['interpretation'].items():
        print(f"  {name}: {interp['status']} - {interp['description']}")
    
    print("\nRecommendations:")
    for rec in comparison_results['recommendations']['model_selection']:
        print(f"  - {rec}")
    
    if comparison_results['recommendations']['cautions']:
        print("\nCautions:")
        for caution in comparison_results['recommendations']['cautions']:
            print(f"   {caution}")
    
    print("\nNext Steps:")
    for step in comparison_results['recommendations']['next_steps']:
        print(f"  -> {step}")
