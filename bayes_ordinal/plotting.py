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
    var_name: str = "alpha",
    hdi_prob: float = 0.94,
    figsize: Tuple[float, float] = (8, 6)
) -> None:
    """
    Plot cutpoint estimates with uncertainty.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from fitted model.
    var_name : str
        Name of cutpoint variable.
    hdi_prob : float
        Width of HDI interval.
    figsize : tuple
        Figure size.
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get cutpoint samples - handle variable name variations
        try:
            cutpoints = idata.posterior[var_name].values
        except KeyError:
            # Try to find the variable with the right name pattern
            available_vars = list(idata.posterior.data_vars.keys())
            matching_vars = [v for v in available_vars if var_name in v]
            if matching_vars:
                cutpoints = idata.posterior[matching_vars[0]].values
                print(f"Using cutpoint variable: {matching_vars[0]}")
            else:
                print(f"Warning: Could not find cutpoint variable '{var_name}'")
                print(f"Available variables: {available_vars}")
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
    var_name: str = "beta",
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
    var_name : str
        Name of coefficient variable.
    feature_names : list, optional
        Names of features for labeling.
    hdi_prob : float
        Width of HDI interval.
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get coefficient samples
    coefs = idata.posterior[var_name].values
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
    var_name: str = "beta",
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
    var_name : str
        Name of variable to plot.
    feature_names : list, optional
        Names of features for labeling.
    hdi_prob : float, default=0.89
        Width of HDI interval (McElreath uses 89%).
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get samples
    samples = idata.posterior[var_name].values
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
    parameter: str = "beta",
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
    parameter : str, default="beta"
        Parameter to contrast.
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
    
    # Get parameter samples
    samples = idata.posterior[parameter].values
    
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
            print("✓ Graphviz visualization displayed successfully!")
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
                print("✓ Graphviz visualization displayed successfully!")
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
    var_name: str = "y_obs",
    n_samples: int = 1000,
    figsize: Tuple[float, float] = (10, 6)
) -> None:
    """
    Plot predicted category probabilities for new data.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from fitted model.
    model : pm.Model
        PyMC model object.
    X_new : np.ndarray
        New feature values for prediction.
    var_name : str
        Name of response variable.
    n_samples : int
        Number of posterior samples to use.
    figsize : tuple
        Figure size.
    """
    try:
        import pymc as pm
        
        # Sample posterior predictive
        with model:
            ppc = pm.sample_posterior_predictive(
                idata,
                var_names=[var_name]
            )
        
        # Get predictions with error handling
        try:
            pred_samples = ppc.posterior_predictive[var_name].values
        except KeyError:
            # Try to find the variable with the right name pattern
            available_vars = list(ppc.posterior_predictive.data_vars.keys())
            matching_vars = [v for v in available_vars if "y_obs" in v]
            if matching_vars:
                pred_samples = ppc.posterior_predictive[matching_vars[0]].values
                print(f"Using predictive variable: {matching_vars[0]}")
            else:
                print(f"Warning: Could not find {var_name} in posterior predictive data")
                print(f"Available variables: {available_vars}")
                return
        
        # Check if we have valid data
        if pred_samples.size == 0:
            print("Warning: No prediction samples available")
            return
        
        # Reshape to (samples, observations) by stacking chains and draws
        n_chains, n_draws, n_obs = pred_samples.shape
        pred_samples_reshaped = pred_samples.reshape(-1, n_obs)  # (chains*draws, obs)
        actual_n_samples = pred_samples_reshaped.shape[0]
        
        # Determine number of categories from the data
        if pred_samples.dtype.kind in 'iuf':  # integer, unsigned integer, float
            K = int(pred_samples.max()) + 1
        else:
            # If not numeric, try to infer from unique values
            unique_vals = np.unique(pred_samples)
            K = len(unique_vals)
        
        # Validate K
        if K <= 0 or K > 20:  # Reasonable bounds
            print(f"Warning: Invalid number of categories K={K}")
            return
        
        # Compute category probabilities
        prob_samples = np.zeros((actual_n_samples, n_obs, K))
        for i in range(actual_n_samples):
            for j in range(n_obs):
                # pred_samples_reshaped[i, j] is already a scalar
                pred_val = int(pred_samples_reshaped[i, j])
                if 0 <= pred_val < K:  # Bounds check
                    counts = np.bincount([pred_val], minlength=K)
                    prob_samples[i, j] = counts / counts.sum()
        
        # Compute mean and HDI
        mean_probs = prob_samples.mean(axis=0)
        hdi_low = np.percentile(prob_samples, 2.5, axis=0)
        hdi_high = np.percentile(prob_samples, 97.5, axis=0)
        
        # Check for valid probabilities
        if np.any(np.isnan(mean_probs)) or np.any(np.isnan(hdi_low)) or np.any(np.isnan(hdi_high)):
            print("Warning: Invalid probability values detected")
            return
        
        # Plot
        fig, axes = plt.subplots(1, K, figsize=figsize)
        if K == 1:
            axes = [axes]
        
        for k in range(K):
            ax = axes[k]
            ax.fill_between(
                range(n_obs), hdi_low[:, k], hdi_high[:, k],
                alpha=0.3, label='95% HDI'
            )
            ax.plot(range(n_obs), mean_probs[:, k], 'o-', label='Mean')
            ax.set_title(f'Category {k}')
            ax.set_xlabel('Observation')
            ax.set_ylabel('Probability')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
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
    var_name: str,
    prior_samples: np.ndarray,
    figsize: Tuple[float, float] = (10, 4)
) -> None:
    """
    Plot prior vs posterior distributions.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from fitted model.
    var_name : str
        Name of variable to plot.
    prior_samples : np.ndarray
        Prior samples for comparison.
    figsize : tuple
        Figure size.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get posterior samples
    post_samples = idata.posterior[var_name].values.flatten()
    
    # Plot histograms
    ax1.hist(prior_samples, bins=30, alpha=0.7, label='Prior', density=True)
    ax1.hist(post_samples, bins=30, alpha=0.7, label='Posterior', density=True)
    ax1.set_xlabel(var_name)
    ax1.set_ylabel('Density')
    ax1.set_title('Prior vs Posterior')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot KDE
    az.plot_dist(prior_samples, ax=ax2, label='Prior')
    az.plot_dist(post_samples, ax=ax2, label='Posterior')
    ax2.set_xlabel(var_name)
    ax2.set_ylabel('Density')
    ax2.set_title('Prior vs Posterior (KDE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    show_and_save("prior_posterior")

# Note: Prior predictive plotting is handled by run_prior_predictive() in workflow/prior_predictive.py
# This function is kept for backward compatibility but delegates to the workflow function
def plot_prior_predictive(
    idata: az.InferenceData,
    y_obs: np.ndarray,
    var_name: str = "y_obs",
    figsize: Tuple[float, float] = (12, 5)
) -> None:
    """
    Plot prior predictive checks (delegates to workflow function).
    
    Note: This function is kept for backward compatibility.
    The actual plotting is handled by run_prior_predictive() in workflow/prior_predictive.py
    which provides more comprehensive prior predictive analysis.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data with prior predictive samples.
    y_obs : np.ndarray
        Observed response variable.
    var_name : str, default="y_obs"
        Name of the response variable in idata.
    figsize : tuple
        Figure size (ignored, uses workflow defaults).
    """
    print("Note: Prior predictive plotting is handled by run_prior_predictive() workflow function")
    print("Use bo.run_prior_predictive(model, plot=True) for comprehensive prior predictive analysis")

# Note: Posterior predictive plotting is handled by run_posterior_predictive() in workflow/posterior_predictive.py
# This function is kept for backward compatibility but delegates to the workflow function
def plot_posterior_predictive(
    ppc: az.InferenceData,
    y_obs: np.ndarray,
    var_name: str = "y_obs",
    figsize: Tuple[float, float] = (12, 5)
) -> None:
    """
    Plot posterior predictive checks (delegates to workflow function).
    
    Note: This function is kept for backward compatibility.
    The actual plotting is handled by run_posterior_predictive() in workflow/posterior_predictive.py
    which provides more comprehensive posterior predictive analysis.
    
    Parameters
    ----------
    ppc : az.InferenceData
        Posterior predictive samples.
    y_obs : np.ndarray
        Observed response variable.
    var_name : str, default="y_obs"
        Name of the response variable in ppc.
    figsize : tuple
        Figure size (ignored, uses workflow defaults).
    """
    print("Note: Posterior predictive plotting is handled by run_posterior_predictive() workflow function")
    print("Use bo.run_posterior_predictive(model, idata, kind='proportions') for comprehensive posterior predictive analysis")
