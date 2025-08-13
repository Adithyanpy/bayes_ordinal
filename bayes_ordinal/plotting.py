"""
Specialized plotting functions for Bayesian ordinal regression.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pymc as pm
from typing import Optional, Tuple, List, Dict, Any

# Configure matplotlib to handle warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='arviz')

from .plot_utils import show_and_save

def plot_cutpoints(
    idata,
    var_name: str = None,
    hdi_prob: float = 0.94,
    figsize: Tuple[float, float] = (8, 6)
) -> None:
    """
    Plot cutpoint estimates with uncertainty.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from fitted model.
    var_name : str, optional
        Name of cutpoint variable. If None, will auto-detect.
    hdi_prob : float
        Width of HDI interval.
    figsize : tuple
        Figure size.
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Auto-detect cutpoint variable if not specified
        if var_name is None:
            available_vars = list(idata.posterior.data_vars.keys())
            # Look for common cutpoint variable names
            cutpoint_patterns = ['cutpoints', 'cuts', 'thresholds', 'alpha']
            var_name = None
            for pattern in cutpoint_patterns:
                matching_vars = [v for v in available_vars if pattern in v.lower()]
                if matching_vars:
                    var_name = matching_vars[0]
                    break
            
            if var_name is None:
                # Fallback: look for variables with appropriate shape
                for var in available_vars:
                    var_data = idata.posterior[var].values
                    if var_data.ndim >= 2 and var_data.shape[-1] > 1:
                        var_name = var
                        break
        
        if var_name is None:
            print("Warning: Could not auto-detect cutpoint variable")
            print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
            return
        
        # Get cutpoint samples
        try:
            cutpoints = idata.posterior[var_name].values
        except KeyError:
            print(f"Warning: Could not find cutpoint variable '{var_name}'")
            print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
            return
        
        n_cuts = cutpoints.shape[-1]
        
        # Plot each cutpoint
        for i in range(n_cuts):
            cut_samples = cutpoints[..., i].flatten()
            
            # Compute HDI with error handling
            try:
                hdi_low, hdi_high = az.hdi(cut_samples, hdi_prob=hdi_prob)
                mean_val = cut_samples.mean()
                
                # Plot point and interval
                ax.errorbar(
                    i, mean_val,
                    yerr=[[mean_val - hdi_low], [hdi_high - mean_val]],
                    fmt='o', capsize=5, capthick=2, markersize=8,
                    color='darkblue', alpha=0.8
                )
            except Exception as e:
                print(f"Error plotting cutpoint {i}: {e}")
                # Fallback to simple mean
                mean_val = cut_samples.mean()
                ax.plot(i, mean_val, 'o', color='darkblue', alpha=0.8)
        
        ax.set_xlabel('Cutpoint Index')
        ax.set_ylabel('Cutpoint Value')
        ax.set_title(f'Cutpoint Estimates ({int(hdi_prob*100)}% HDI)')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(n_cuts))
        ax.set_xticklabels([f'α_{i+1}' for i in range(n_cuts)])
        show_and_save("cutpoints")
        
    except Exception as e:
        print(f"Error in plot_cutpoints: {e}")
        print("Skipping cutpoint visualization")

def plot_coefficient_effects(
    idata,
    var_name: str = None,
    feature_names: Optional[List[str]] = None,
    hdi_prob: float = 0.94,
    figsize: Tuple[float, float] = (10, 6)
) -> None:
    """
    Plot coefficient effects with uncertainty intervals.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from fitted model.
    var_name : str, optional
        Name of coefficient variable. If None, will auto-detect.
    feature_names : list, optional
        Names of features for labeling.
    hdi_prob : float
        Width of HDI interval.
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Auto-detect coefficient variable if not specified
    if var_name is None:
        available_vars = list(idata.posterior.data_vars.keys())
        # Look for common coefficient variable names
        coef_patterns = ['beta', 'coefficients', 'coef', 'slopes']
        var_name = None
        for pattern in coef_patterns:
            matching_vars = [v for v in available_vars if pattern in v.lower()]
            if matching_vars:
                var_name = matching_vars[0]
                break
        
        if var_name is None:
            # Fallback: look for variables with appropriate shape
            for var in available_vars:
                var_data = idata.posterior[var].values
                if var_data.ndim >= 2 and var_data.shape[-1] > 1:
                    var_name = var
                    break
    
    if var_name is None:
        print("Warning: Could not auto-detect coefficient variable")
        print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
        return
    
    # Get coefficient samples
    try:
        coefs = idata.posterior[var_name].values
    except KeyError:
        print(f"Warning: Could not find coefficient variable '{var_name}'")
        print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
        return
    
    n_features = coefs.shape[-1]
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]
    else:
        # Ensure feature_names are strings
        feature_names = [str(name) for name in feature_names]
    
    # Compute statistics for each coefficient
    means = []
    hdi_lows = []
    hdi_highs = []
    
    for i in range(n_features):
        coef_samples = coefs[..., i].flatten()
        hdi_low, hdi_high = az.hdi(coef_samples, hdi_prob=hdi_prob)
        mean_val = coef_samples.mean()
        
        means.append(mean_val)
        hdi_lows.append(hdi_low)
        hdi_highs.append(hdi_high)
    
    # Create forest plot
    y_pos = np.arange(n_features)
    yerr = [
        np.array(means) - np.array(hdi_lows),
        np.array(hdi_highs) - np.array(means)
    ]
    
    ax.errorbar(
        means, y_pos,
        xerr=yerr,
        fmt='o', capsize=5, capthick=2, markersize=8
    )
    
    # Add zero line
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='No effect')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Coefficient Value')
    ax.set_title(f'Coefficient Effects ({int(hdi_prob*100)}% HDI)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    show_and_save("coefficient_effects")

def plot_forest(
    idata,
    var_name: str = None,
    feature_names: Optional[List[str]] = None,
    hdi_prob: float = 0.89,  # McElreath's preferred HDI
    figsize: Tuple[float, float] = (10, 6)
) -> None:
    """
    Plot McElreath-style forest plot with enhanced styling.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from fitted model.
    var_name : str, optional
        Name of variable to plot. If None, will auto-detect.
    feature_names : list, optional
        Names of features for labeling.
    hdi_prob : float, default=0.89
        Width of HDI interval (McElreath uses 89%).
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Auto-detect variable if not specified
    if var_name is None:
        available_vars = list(idata.posterior.data_vars.keys())
        # Look for common variable names
        var_patterns = ['beta', 'coefficients', 'cutpoints', 'gamma', 'alpha']
        var_name = None
        for pattern in var_patterns:
            matching_vars = [v for v in available_vars if pattern in v.lower()]
            if matching_vars:
                var_name = matching_vars[0]
                break
        
        if var_name is None:
            # Fallback: use first variable with appropriate shape
            for var in available_vars:
                var_data = idata.posterior[var].values
                if var_data.ndim >= 2 and var_data.shape[-1] > 1:
                    var_name = var
                    break
    
    if var_name is None:
        print("Warning: Could not auto-detect variable to plot")
        print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
        return
    
    # Get samples
    try:
        samples = idata.posterior[var_name].values
    except KeyError:
        print(f"Warning: Could not find variable '{var_name}'")
        print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
        return
    
    n_features = samples.shape[-1]
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]
    else:
        # Ensure feature_names are strings
        feature_names = [str(name) for name in feature_names]
    
    # Compute statistics
    means = []
    hdi_lows = []
    hdi_highs = []
    
    for i in range(n_features):
        feature_samples = samples[..., i].flatten()
        try:
            hdi_low, hdi_high = az.hdi(feature_samples, hdi_prob=hdi_prob)
            mean_val = feature_samples.mean()
            
            means.append(mean_val)
            hdi_lows.append(hdi_low)
            hdi_highs.append(hdi_high)
        except Exception as e:
            print(f"Could not compute HDI for feature {i}: {e}")
            mean_val = feature_samples.mean()
            hdi_low = np.percentile(feature_samples, (1-hdi_prob)/2 * 100)
            hdi_high = np.percentile(feature_samples, (1+hdi_prob)/2 * 100)
            
            means.append(mean_val)
            hdi_lows.append(hdi_low)
            hdi_highs.append(hdi_high)
    
    # McElreath-style forest plot
    y_pos = np.arange(n_features)
    
    # Plot HDI intervals
    ax.hlines(y_pos, hdi_lows, hdi_highs, linewidth=3, alpha=0.7, color='steelblue')
    
    # Plot means
    ax.plot(means, y_pos, 'o', markersize=8, color='darkblue', markeredgecolor='white', markeredgewidth=2)
    
    # Add zero line
    ax.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='No effect')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Parameter Value')
    ax.set_title(f'McElreath-Style Forest Plot ({int(hdi_prob*100)}% HDI)')
    ax.grid(True, alpha=0.2, axis='x')
    ax.legend()
    
    # McElreath-style formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    show_and_save("forest_plot")

def plot_contrast(
    idata,
    group1_indices: np.ndarray,
    group2_indices: np.ndarray,
    parameter: str = None,
    group_names: Optional[List[str]] = None,
    hdi_prob: float = 0.89,
    figsize: Tuple[float, float] = (12, 5)
) -> Dict[str, Any]:
    """
    Perform McElreath-style contrast analysis between two groups.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from fitted model.
    group1_indices : np.ndarray
        Indices for first group.
    group2_indices : np.ndarray
        Indices for second group.
    parameter : str, optional
        Parameter to contrast. If None, will auto-detect.
    group_names : list, optional
        Names for the groups.
    hdi_prob : float, default=0.89
        Width of HDI interval.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    dict
        Contrast analysis results.
    """
    if group_names is None:
        group_names = ["Group 1", "Group 2"]
    
    # Auto-detect parameter if not specified
    if parameter is None:
        available_vars = list(idata.posterior.data_vars.keys())
        # Look for common parameter names
        param_patterns = ['beta', 'coefficients', 'cutpoints', 'gamma', 'alpha']
        parameter = None
        for pattern in param_patterns:
            matching_vars = [v for v in available_vars if pattern in v.lower()]
            if matching_vars:
                parameter = matching_vars[0]
                break
        
        if parameter is None:
            # Fallback: use first variable with appropriate shape
            for var in available_vars:
                var_data = idata.posterior[var].values
                if var_data.ndim >= 2 and var_data.shape[-1] > 1:
                    parameter = var
                    break
    
    if parameter is None:
        print("Warning: Could not auto-detect parameter for contrast analysis")
        print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
        return {}
    
    # Get parameter samples
    try:
        samples = idata.posterior[parameter].values
    except KeyError:
        print(f"Warning: Could not find parameter '{parameter}'")
        print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
        return {}
    
    # Extract group-specific samples
    group1_samples = samples[..., group1_indices, :]
    group2_samples = samples[..., group2_indices, :]
    
    # Compute group means
    group1_means = group1_samples.mean(axis=-2)  # Average across group members
    group2_means = group2_samples.mean(axis=-2)
    
    # Compute contrast (difference)
    contrast_samples = group1_means - group2_means
    
    # Compute statistics
    contrast_mean = contrast_samples.mean(axis=(0, 1))
    
    # Ensure contrast_mean is an array
    if np.isscalar(contrast_mean):
        contrast_mean = np.array([contrast_mean])
    
    contrast_hdi_result = az.hdi(contrast_samples, hdi_prob=hdi_prob)
    
    # Handle different HDI result formats
    if contrast_hdi_result.ndim == 1:
        contrast_hdi_low, contrast_hdi_high = contrast_hdi_result[0], contrast_hdi_result[1]
    elif contrast_hdi_result.ndim == 2:
        contrast_hdi_low, contrast_hdi_high = contrast_hdi_result[0], contrast_hdi_result[1]
    else:
        # For multi-dimensional arrays, take the first two elements
        contrast_hdi_low = contrast_hdi_result[0] if contrast_hdi_result.size > 0 else 0
        contrast_hdi_high = contrast_hdi_result[1] if contrast_hdi_result.size > 1 else 0
    
    # Ensure HDI results are arrays
    if np.isscalar(contrast_hdi_low):
        contrast_hdi_low = np.array([contrast_hdi_low])
        contrast_hdi_high = np.array([contrast_hdi_high])
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Group comparison
    group1_mean = group1_means.mean(axis=(0, 1))
    group2_mean = group2_means.mean(axis=(0, 1))
    
    # Ensure we have arrays for plotting
    if np.isscalar(group1_mean):
        group1_mean = np.array([group1_mean])
        group2_mean = np.array([group2_mean])
    
    group1_hdi_result = az.hdi(group1_means, hdi_prob=hdi_prob)
    group2_hdi_result = az.hdi(group2_means, hdi_prob=hdi_prob)
    
    # Handle different HDI result formats for group1
    if group1_hdi_result.ndim == 1:
        group1_hdi_low, group1_hdi_high = group1_hdi_result[0], group1_hdi_result[1]
    elif group1_hdi_result.ndim == 2:
        group1_hdi_low, group1_hdi_high = group1_hdi_result[0], group1_hdi_result[1]
    else:
        group1_hdi_low = group1_hdi_result[0] if group1_hdi_result.size > 0 else 0
        group1_hdi_high = group1_hdi_result[1] if group1_hdi_result.size > 1 else 0
        
    # Handle different HDI result formats for group2
    if group2_hdi_result.ndim == 1:
        group2_hdi_low, group2_hdi_high = group2_hdi_result[0], group2_hdi_result[1]
    elif group2_hdi_result.ndim == 2:
        group2_hdi_low, group2_hdi_high = group2_hdi_result[0], group2_hdi_result[1]
    else:
        group2_hdi_low = group2_hdi_result[0] if group2_hdi_result.size > 0 else 0
        group2_hdi_high = group2_hdi_result[1] if group2_hdi_result.size > 1 else 0
    
    # Ensure HDI results are arrays
    if np.isscalar(group1_hdi_low):
        group1_hdi_low = np.array([group1_hdi_low])
        group1_hdi_high = np.array([group1_hdi_high])
    if np.isscalar(group2_hdi_low):
        group2_hdi_low = np.array([group2_hdi_low])
        group2_hdi_high = np.array([group2_hdi_high])
    
    y_pos = np.arange(len(group1_mean))
    
    # Simplify plotting to avoid complex array shape issues
    for i in range(len(group1_mean)):
        # Group comparison
        ax1.errorbar(group1_mean[i], i, 
                    xerr=[[abs(group1_mean[i] - group1_hdi_low[i])], [abs(group1_hdi_high[i] - group1_mean[i])]],
                    fmt='o', label=group_names[0] if i == 0 else "", capsize=5)
        ax1.errorbar(group2_mean[i], i, 
                    xerr=[[abs(group2_mean[i] - group2_hdi_low[i])], [abs(group2_hdi_high[i] - group2_mean[i])]],
                    fmt='s', label=group_names[1] if i == 0 else "", capsize=5)
        
        # Contrast plot
        ax2.errorbar(contrast_mean[i], i, 
                    xerr=[[abs(contrast_mean[i] - contrast_hdi_low[i])], [abs(contrast_hdi_high[i] - contrast_mean[i])]],
                    fmt='o', capsize=5, color='red')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f'Param {i}' for i in range(len(group1_mean))])
    ax1.set_xlabel('Parameter Value')
    ax1.set_title('Group Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.axvline(0, color='black', linestyle='--', alpha=0.7, label='No difference')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'Param {i}' for i in range(len(contrast_mean))])
    ax2.set_xlabel('Contrast (Group 1 - Group 2)')
    ax2.set_title('Contrast Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    show_and_save("contrast_analysis")
    
    # Return results
    results = {
        'group1_mean': group1_mean,
        'group2_mean': group2_mean,
        'contrast_mean': contrast_mean,
        'contrast_hdi_low': contrast_hdi_low,
        'contrast_hdi_high': contrast_hdi_high,
        'hdi_prob': hdi_prob
    }
    
    return results

def plot_model_structure(model: pm.Model) -> None:
    """
    Plot model structure using PyMC's built-in graphviz visualization.
    
    Parameters
    ----------
    model : pm.Model
        PyMC model to visualize.
    """
    try:
        graph = pm.model_to_graphviz(model)
        print("Model Structure (PyMC Graphviz):")
        # Try to render the graph if possible
        try:
            # Set graph attributes for proper sizing
            graph.attr(rankdir='TB', size='8,6', dpi='300', nodesep='0.5', ranksep='0.5')
            # Use graphviz's built-in display functionality with controlled size
            graph.render(view=True, format='png', engine='dot', cleanup=True)
            print(" Graphviz visualization displayed successfully!")
        except Exception as render_error:
            # If rendering fails, just print the DOT representation
            print(graph)
            print(f"\nNote: Could not render graphviz visualization: {render_error}")
            print("The DOT representation above shows the model structure.")
        return graph
    except Exception as e:
        print(f"Error with PyMC graphviz: {e}")
        print("Using alternative visualization...")
        print("Model Variables:")
        for var in model.free_RVs:
            print(f"  {var.name}: {type(var).__name__}")
        print("Model Deterministics:")
        for var in model.deterministics:
            print(f"  {var.name}: {type(var).__name__}")

def draw_causal_graph(
    model: Optional[pm.Model] = None,
    model_type: str = "ordinal",
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 6)
) -> None:
    """
    Draw causal graph for ordinal regression models using PyMC's built-in graphviz.
    
    Parameters
    ----------
    model : pm.Model, optional
        PyMC model to visualize. If provided, uses pm.model_to_graphviz(model).
    model_type : str, default="ordinal"
        Type of model ("ordinal", "hierarchical", "continuous"). Used only if model is None.
    feature_names : list, optional
        Names of features for labeling. Used only if model is None.
    figsize : tuple
        Figure size. Used only if model is None.
    """
    if model is not None:
        # Use PyMC's built-in model visualization
        try:
            graph = pm.model_to_graphviz(model)
            print("Model Structure (PyMC Graphviz):")
            # Try to render the graph if possible
            try:
                # Set graph attributes for proper sizing
                graph.attr(rankdir='TB', size='8,6', dpi='300', nodesep='0.5', ranksep='0.5')
                # Use graphviz's built-in display functionality with controlled size
                graph.render(view=True, format='png', engine='dot', cleanup=True)
                print(" Graphviz visualization displayed successfully!")
            except Exception as render_error:
                # If rendering fails, just print the DOT representation
                print(graph)
                print(f"\nNote: Could not render graphviz visualization: {render_error}")
                print("The DOT representation above shows the model structure.")
            return graph
        except Exception as e:
            print(f"Error with PyMC graphviz: {e}")
            print("Falling back to custom visualization...")
    
    # Fallback to custom visualization if no model provided or graphviz fails
    fig, ax = plt.subplots(figsize=figsize)
    
    if model_type == "ordinal":
        if feature_names is None:
            feature_names = ["X₁", "X₂", "X₃"]
        
        # Draw nodes
        node_positions = {
            'features': [(0, 0.8), (0, 0.4), (0, 0)],
            'linear': (2, 0.4),
            'cutpoints': [(4, 0.8), (4, 0.4), (4, 0)],
            'y': (6, 0.4)
        }
        
        # Draw feature nodes
        for i, (name, pos) in enumerate(zip(feature_names, node_positions['features'])):
            ax.text(pos[0], pos[1], name, ha='center', va='center', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                   fontsize=12)
        
        # Draw linear predictor
        ax.text(node_positions['linear'][0], node_positions['linear'][1], 'μ', 
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
               fontsize=14)
        
        # Draw cutpoints
        for i, pos in enumerate(node_positions['cutpoints']):
            ax.text(pos[0], pos[1], f'α_{i+1}', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7),
                   fontsize=12)
        
        # Draw outcome
        ax.text(node_positions['y'][0], node_positions['y'][1], 'Y', 
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
               fontsize=14)
        
        # Draw arrows with proper positioning to avoid going inside boxes
        for pos in node_positions['features']:
            # Arrow from feature to linear predictor (right edge to left edge)
            ax.annotate('', xy=(node_positions['linear'][0] - 0.3, node_positions['linear'][1]), 
                       xytext=(pos[0] + 0.3, pos[1]),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        for pos in node_positions['cutpoints']:
            # Arrow from cutpoint to outcome (left edge to right edge)
            ax.annotate('', xy=(node_positions['y'][0] - 0.3, node_positions['y'][1]), 
                       xytext=(pos[0] + 0.3, pos[1]),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Arrow from linear predictor to outcome (right edge to left edge)
        ax.annotate('', xy=(node_positions['y'][0] - 0.3, node_positions['y'][1]), 
                   xytext=(node_positions['linear'][0] + 0.3, node_positions['linear'][1]),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        ax.set_xlim(-1, 7)
        ax.set_ylim(-0.5, 1.3)
        ax.set_title('Causal Graph: Ordinal Regression Model', fontsize=14, fontweight='bold')
        
    elif model_type == "hierarchical":
        # Hierarchical model causal graph
        node_positions = {
            'group': (0, 0.5),
            'features': [(2, 0.8), (2, 0.2)],
            'linear': (4, 0.5),
            'y': (6, 0.5)
        }
        
        # Draw group node
        ax.text(node_positions['group'][0], node_positions['group'][1], 'Group', 
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
               fontsize=12)
        
        # Draw feature nodes
        for i, pos in enumerate(node_positions['features']):
            ax.text(pos[0], pos[1], f'X_{i+1}', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   fontsize=12)
        
        # Draw linear predictor
        ax.text(node_positions['linear'][0], node_positions['linear'][1], 'μ', 
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7),
               fontsize=14)
        
        # Draw outcome
        ax.text(node_positions['y'][0], node_positions['y'][1], 'Y', 
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
               fontsize=14)
        
        # Draw arrows with proper positioning to avoid going inside boxes
        # Arrow from group to linear predictor (right edge to left edge)
        ax.annotate('', xy=(node_positions['linear'][0] - 0.3, node_positions['linear'][1]), 
                   xytext=(node_positions['group'][0] + 0.3, node_positions['group'][1]),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        for pos in node_positions['features']:
            # Arrow from feature to linear predictor (right edge to left edge)
            ax.annotate('', xy=(node_positions['linear'][0] - 0.3, node_positions['linear'][1]), 
                       xytext=(pos[0] + 0.3, pos[1]),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Arrow from linear predictor to outcome (right edge to left edge)
        ax.annotate('', xy=(node_positions['y'][0] - 0.3, node_positions['y'][1]), 
                   xytext=(node_positions['linear'][0] + 0.3, node_positions['linear'][1]),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        ax.set_xlim(-1, 7)
        ax.set_ylim(-0.5, 1.3)
        ax.set_title('Causal Graph: Hierarchical Model', fontsize=14, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    show_and_save("causal_graph")

def plot_category_probabilities(
    idata,
    model,
    X_new: np.ndarray,
    var_name: str = None,
    n_samples: int = 1000,
    figsize: Tuple[float, float] = (10, 6)
) -> None:
    """
    Plot category probabilities for new data points.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from fitted model.
    model : pm.Model
        The fitted PyMC model.
    X_new : np.ndarray
        New feature values for prediction.
    var_name : str, optional
        Name of response variable. If None, will auto-detect.
    n_samples : int
        Number of posterior samples to use.
    figsize : tuple
        Figure size.
    """
    try:
        # Auto-detect response variable if not specified
        if var_name is None:
            available_vars = list(idata.posterior.data_vars.keys())
            # Look for common response variable names
            response_patterns = ['y', 'response', 'outcome', 'target']
            var_name = None
            for pattern in response_patterns:
                matching_vars = [v for v in available_vars if pattern in v.lower()]
                if matching_vars:
                    var_name = matching_vars[0]
                    break
            
            if var_name is None:
                # Fallback: look for variables with appropriate shape
                for var in available_vars:
                    var_data = idata.posterior[var].values
                    if var_data.ndim >= 2 and var_data.shape[-1] > 1:
                        var_name = var
                        break
        
        if var_name is None:
            print("Warning: Could not auto-detect response variable")
            print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
            return
        
        # Generate posterior predictive samples
        with model:
            ppc = pm.sample_posterior_predictive(
                idata,
                samples=n_samples,
                var_names=[var_name],
                return_inferencedata=False
            )
        
        # Extract predictions
        if var_name in ppc:
            predictions = ppc[var_name]
        else:
            # Try to find the prediction variable
            pred_vars = list(ppc.keys())
            if pred_vars:
                predictions = ppc[pred_vars[0]]
                print(f"Using prediction variable: {pred_vars[0]}")
            else:
                print("No prediction variables found")
                return
        
        # Calculate category probabilities
        n_categories = len(np.unique(predictions))
        n_points = X_new.shape[0]
        
        # Initialize probability matrix
        prob_matrix = np.zeros((n_points, n_categories))
        
        for i in range(n_points):
            point_predictions = predictions[:, i]
            for j in range(n_categories):
                prob_matrix[i, j] = np.mean(point_predictions == j)
        
        # Plot results
        fig, ax = plt.subplots(figsize=figsize)
        
        x_pos = np.arange(n_points)
        width = 0.8 / n_categories
        
        for j in range(n_categories):
            ax.bar(x_pos + j * width, prob_matrix[:, j], width, 
                  label=f'Category {j}', alpha=0.8)
        
        ax.set_xlabel('Data Point')
        ax.set_ylabel('Probability')
        ax.set_title('Category Probabilities for New Data')
        ax.set_xticks(x_pos + width * (n_categories - 1) / 2)
        ax.set_xticklabels([f'Point {i+1}' for i in range(n_points)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        show_and_save("category_probabilities")
        
    except Exception as e:
        print(f"Error in plot_category_probabilities: {e}")
        print("Skipping category probability visualization")

def plot_model_comparison(
    idatas: Dict[str, Any],
    ic: str = "loo",
    figsize: Tuple[float, float] = (10, 6)
) -> None:
    """
    Plot model comparison using information criteria.
    
    Parameters
    ----------
    idatas : dict
        Dictionary mapping model names to inference data.
    ic : str
        Information criterion to use ("loo" or "waic").
    figsize : tuple
        Figure size.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Compute IC values
    ic_values = []
    ic_ses = []
    model_names = []
    
    for name, idata in idatas.items():
        if ic == "loo":
            loo_result = az.loo(idata)
            ic_values.append(loo_result.elpd_loo)
            ic_ses.append(loo_result.se)
        else:
            waic_result = az.waic(idata)
            ic_values.append(waic_result.elpd_waic)
            ic_ses.append(waic_result.se)
        model_names.append(name)
    
    # Plot IC values
    y_pos = np.arange(len(model_names))
    ax1.errorbar(
        ic_values, y_pos,
        xerr=ic_ses,
        fmt='o', capsize=5, capthick=2, markersize=8
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_names)
    ax1.set_xlabel(f'{ic.upper()} Value')
    ax1.set_title(f'Model Comparison ({ic.upper()})')
    ax1.grid(True, alpha=0.3)
    
    # Plot differences from best model
    ic_values = np.array(ic_values)
    best_idx = np.argmax(ic_values)
    differences = ic_values - ic_values[best_idx]
    
    ax2.barh(y_pos, differences)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names)
    ax2.set_xlabel(f'{ic.upper()} Difference from Best')
    ax2.set_title('Relative Performance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    show_and_save("model_comparison")

# Note: Model comparison interpretation plotting is handled by plot_model_comparison_interpretation() in workflow/cross_validation.py
# This function is kept for backward compatibility but delegates to the workflow function
def plot_model_comparison_interpretation(
    idatas: Dict[str, Any],
    ic: str = "loo",
    figsize: Tuple[float, float] = (12, 8)
) -> Dict[str, Any]:
    """
    Plot McElreath-style model comparison with interpretation rules (delegates to workflow function).
    
    Note: This function is kept for backward compatibility.
    The actual plotting is handled by plot_model_comparison_interpretation() in workflow/cross_validation.py
    which provides more comprehensive model comparison analysis with detailed interpretation.
    
    Parameters
    ----------
    idatas : dict
        Dictionary mapping model names to inference data.
    ic : str, default="loo"
        Information criterion to use ("loo" or "waic").
    figsize : tuple
        Figure size (ignored, uses workflow defaults).
        
    Returns
    -------
    dict
        Model comparison results with McElreath interpretation.
    """
    print("Note: Model comparison interpretation plotting is handled by workflow function")
    print("Use bo.compare_models_interpretation(models, idatas) for comprehensive model comparison analysis")
    
    # Return empty dict for backward compatibility
    return {
        'ic_values': [],
        'ic_ses': [],
        'differences': [],
        'best_model': 'unknown',
        'interpretation': {},
        'ic_results': {}
    }

def plot_prior_posterior(
    idata,
    var_name: str = None,
    prior_samples: np.ndarray = None,
    figsize: Tuple[float, float] = (10, 4)
) -> None:
    """
    Plot prior vs posterior distributions.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from fitted model.
    var_name : str, optional
        Name of variable to plot. If None, will auto-detect.
    prior_samples : np.ndarray, optional
        Prior samples for comparison.
    figsize : tuple
        Figure size.
    """
    try:
        # Auto-detect variable if not specified
        if var_name is None:
            available_vars = list(idata.posterior.data_vars.keys())
            # Look for common variable names
            var_patterns = ['beta', 'coefficients', 'cutpoints', 'gamma', 'alpha']
            var_name = None
            for pattern in var_patterns:
                matching_vars = [v for v in available_vars if pattern in v.lower()]
                if matching_vars:
                    var_name = matching_vars[0]
                    break
            
            if var_name is None:
                # Fallback: use first variable with appropriate shape
                for var in available_vars:
                    var_data = idata.posterior[var].values
                    if var_data.ndim >= 2 and var_data.shape[-1] > 1:
                        var_name = var
                        break
        
        if var_name is None:
            print("Warning: Could not auto-detect variable for prior-posterior plot")
            print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
            return
        
        # Get posterior samples
        try:
            posterior_samples = idata.posterior[var_name].values
        except KeyError:
            print(f"Warning: Could not find variable '{var_name}'")
            print(f"Available variables: {list(idata.posterior.data_vars.keys())}")
            return
        
        # Flatten posterior samples
        posterior_flat = posterior_samples.flatten()
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Posterior histogram
        ax1.hist(posterior_flat, bins=50, alpha=0.7, density=True, 
                color='steelblue', edgecolor='black')
        ax1.set_xlabel('Parameter Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Posterior Distribution: {var_name}')
        ax1.grid(True, alpha=0.3)
        
        # Prior vs Posterior comparison
        if prior_samples is not None:
            prior_flat = prior_samples.flatten()
            
            # Plot both distributions
            ax2.hist(prior_flat, bins=50, alpha=0.5, density=True, 
                    color='red', label='Prior', edgecolor='black')
            ax2.hist(posterior_flat, bins=50, alpha=0.5, density=True, 
                    color='blue', label='Posterior', edgecolor='black')
            
            ax2.set_xlabel('Parameter Value')
            ax2.set_ylabel('Density')
            ax2.set_title('Prior vs Posterior Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Just plot posterior if no prior provided
            ax2.hist(posterior_flat, bins=50, alpha=0.7, density=True, 
                    color='steelblue', edgecolor='black')
            ax2.set_xlabel('Parameter Value')
            ax2.set_ylabel('Density')
            ax2.set_title(f'Posterior Distribution: {var_name}')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        show_and_save("prior_posterior")
        
    except Exception as e:
        print(f"Error in plot_prior_posterior: {e}")
        print("Skipping prior-posterior visualization")

def plot_prior_predictive(
    idata: az.InferenceData,
    y_obs: np.ndarray,
    var_name: str = None,
    figsize: Tuple[float, float] = (12, 5)
) -> None:
    """
    Plot prior predictive checks.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data with prior predictive samples.
    y_obs : np.ndarray
        Observed response values.
    var_name : str, optional
        Name of response variable. If None, will auto-detect.
    figsize : tuple
        Figure size.
    """
    try:
        # Auto-detect response variable if not specified
        if var_name is None:
            available_vars = list(idata.prior_predictive.data_vars.keys())
            # Look for common response variable names
            response_patterns = ['y', 'response', 'outcome', 'target']
            var_name = None
            for pattern in response_patterns:
                matching_vars = [v for v in available_vars if pattern in v.lower()]
                if matching_vars:
                    var_name = matching_vars[0]
                    break
            
            if var_name is None:
                # Fallback: use first variable with appropriate shape
                for var in available_vars:
                    var_data = idata.prior_predictive[var].values
                    if var_data.ndim >= 2 and var_data.shape[-1] > 1:
                        var_name = var
                        break
        
        if var_name is None:
            print("Warning: Could not auto-detect response variable for prior predictive plot")
            print(f"Available variables: {list(idata.prior_predictive.data_vars.keys())}")
            return
        
        # Get prior predictive samples
        try:
            prior_pred = idata.prior_predictive[var_name].values
        except KeyError:
            print(f"Warning: Could not find variable '{var_name}' in prior predictive data")
            print(f"Available variables: {list(idata.prior_predictive.data_vars.keys())}")
            return
        
        # Flatten samples
        prior_pred_flat = prior_pred.reshape(-1, prior_pred.shape[-1])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Prior predictive distribution
        ax1.hist(prior_pred_flat.flatten(), bins=50, alpha=0.7, density=True, 
                color='lightblue', edgecolor='black', label='Prior Predictive')
        ax1.hist(y_obs, bins=50, alpha=0.7, density=True, 
                color='red', edgecolor='black', label='Observed')
        ax1.set_xlabel('Response Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Prior Predictive vs Observed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Prior predictive summary statistics
        n_samples = prior_pred_flat.shape[0]
        sample_means = np.mean(prior_pred_flat, axis=1)
        sample_stds = np.std(prior_pred_flat, axis=1)
        
        ax2.scatter(sample_means, sample_stds, alpha=0.6, color='steelblue')
        ax2.axvline(np.mean(y_obs), color='red', linestyle='--', label='Observed Mean')
        ax2.axhline(np.std(y_obs), color='red', linestyle=':', label='Observed Std')
        ax2.set_xlabel('Sample Mean')
        ax2.set_ylabel('Sample Standard Deviation')
        ax2.set_title('Prior Predictive Summary Statistics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        show_and_save("prior_predictive")
        
    except Exception as e:
        print(f"Error in plot_prior_predictive: {e}")
        print("Skipping prior predictive visualization")

def plot_posterior_predictive(
    ppc: az.InferenceData,
    y_obs: np.ndarray,
    var_name: str = None,
    figsize: Tuple[float, float] = (12, 5)
) -> None:
    """
    Plot posterior predictive checks.
    
    Parameters
    ----------
    ppc : az.InferenceData
        Posterior predictive samples.
    y_obs : np.ndarray
        Observed response values.
    var_name : str, optional
        Name of response variable. If None, will auto-detect.
    figsize : tuple
        Figure size.
    """
    try:
        # Auto-detect response variable if not specified
        if var_name is None:
            available_vars = list(ppc.predictions.data_vars.keys())
            # Look for common response variable names
            response_patterns = ['y', 'response', 'outcome', 'target']
            var_name = None
            for pattern in response_patterns:
                matching_vars = [v for v in available_vars if pattern in v.lower()]
                if matching_vars:
                    var_name = matching_vars[0]
                    break
            
            if var_name is None:
                # Fallback: use first variable with appropriate shape
                for var in available_vars:
                    var_data = ppc.predictions[var].values
                    if var_data.ndim >= 2 and var_data.shape[-1] > 1:
                        var_name = var
                        break
        
        if var_name is None:
            print("Warning: Could not auto-detect response variable for posterior predictive plot")
            print(f"Available variables: {list(ppc.predictions.data_vars.keys())}")
            return
        
        # Get posterior predictive samples
        try:
            post_pred = ppc.predictions[var_name].values
        except KeyError:
            print(f"Warning: Could not find variable '{var_name}' in posterior predictive data")
            print(f"Available variables: {list(ppc.predictions.data_vars.keys())}")
            return
        
        # Flatten samples
        post_pred_flat = post_pred.reshape(-1, post_pred.shape[-1])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Posterior predictive distribution
        ax1.hist(post_pred_flat.flatten(), bins=50, alpha=0.7, density=True, 
                color='lightgreen', edgecolor='black', label='Posterior Predictive')
        ax1.hist(y_obs, bins=50, alpha=0.7, density=True, 
                color='red', edgecolor='black', label='Observed')
        ax1.set_xlabel('Response Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Posterior Predictive vs Observed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Posterior predictive summary statistics
        n_samples = post_pred_flat.shape[0]
        sample_means = np.mean(post_pred_flat, axis=1)
        sample_stds = np.std(post_pred_flat, axis=1)
        
        ax2.scatter(sample_means, sample_stds, alpha=0.6, color='darkgreen')
        ax2.axvline(np.mean(y_obs), color='red', linestyle='--', label='Observed Mean')
        ax2.axhline(np.std(y_obs), color='red', linestyle=':', label='Observed Std')
        ax2.set_xlabel('Sample Mean')
        ax2.set_ylabel('Sample Standard Deviation')
        ax2.set_title('Posterior Predictive Summary Statistics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        show_and_save("posterior_predictive")
        
    except Exception as e:
        print(f"Error in plot_posterior_predictive: {e}")
        print("Skipping posterior predictive visualization")
