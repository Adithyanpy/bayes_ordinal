import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt


def run_prior_predictive(
    model: pm.Model,
    draws: int = 200,
    plot: bool = True,
    y_obs: np.ndarray = None,
    model_name: str = None,
    validate_model: bool = True,
    plot_kwargs: dict = None,
    custom_plots: dict = None
) -> az.InferenceData:
    """
    Run prior predictive checks on a PyMC model.

    This function samples from the prior predictive distribution and optionally
    visualizes the distribution of the response variable. It automatically handles
    different model types and provides comprehensive diagnostics.

    Parameters
    ----------
    model : pm.Model
        The unsampled PyMC model containing only prior definitions.
    draws : int, default=200
        Number of prior predictive samples.
    plot : bool, default=True
        Whether to plot histograms of the prior predictive distribution.
    y_obs : np.ndarray, optional
        Observed response variable for comparison with prior predictive.
    model_name : str, optional
        Explicit model name for variable detection. If None, will be inferred.
    validate_model : bool, default=True
        Whether to perform basic model validation before sampling.

    plot_kwargs : dict, optional
        Additional keyword arguments for plotting customization.
    custom_plots : dict, optional
        Dictionary specifying which plots to show. Available options:
        - 'prior_samples': Prior Predictive Samples plot
        - 'mean_distribution': Mean Prior Predictive plot  
        - 'observed': Observed data plot (bars)
        - 'category_counts': Category Counts per Sample plot
        - 'total_observations': Total Observations per Sample plot
        - 'category_proportions': Category Proportions Across Samples plot
        Default: Only mean_distribution and observed plots shown
        
        Example:
        >>> custom_plots = {
        ...     'prior_samples': True,      # Show this plot
        ...     'mean_distribution': True,  # Show this plot
        ...     'observed': True,           # Show this plot
        ...     'category_counts': False,   # Skip this plot
        ...     'total_observations': False, # Skip this plot
        ...     'category_proportions': False # Skip this plot
        ... }

    Returns
    -------
    idata : az.InferenceData
        An ArviZ InferenceData object with prior predictive samples.

    Examples
    --------
    Basic usage:
    >>> model = cumulative_model(y, X, K=4)
    >>> idata = run_prior_predictive(model)
    
    Advanced usage:
    >>> idata = run_prior_predictive(
    ...     model, draws=1000, plot=True, y_obs=y,
    ...     model_name="my_model"
    ... )
    
    Custom plots (only show specific plots):
    >>> custom_plots = {
    ...     'prior_samples': True,
    ...     'mean_distribution': True,
    ...     'observed': True,
    ...     'category_counts': False,      # Skip this plot
    ...     'total_observations': False,  # Skip this plot
    ...     'category_proportions': False # Skip this plot
    ... }
    >>> idata = run_prior_predictive(
    ...     model, draws=1000, plot=True, y_obs=y,
    ...     custom_plots=custom_plots
    ... )
    
    Without plotting:
    >>> idata = run_prior_predictive(model, plot=False)
    """
    # Input validation
    if not isinstance(model, pm.Model):
        raise TypeError("model must be a PyMC Model instance")
    
    if draws <= 0:
        raise ValueError("draws must be positive")
    
    if y_obs is not None and not isinstance(y_obs, np.ndarray):
        y_obs = np.array(y_obs)
    
    # Model validation
    if validate_model:
        _validate_model_structure(model)
    
    # Auto-detect model name if not provided
    if model_name is None:
        model_name = _infer_model_name(model)
    
    print(f" Running prior predictive check for model: {model_name}")
    print(f" Drawing {draws} samples from prior predictive distribution")
    
    # Run prior predictive sampling
    try:
        with model:
            idata = pm.sample_prior_predictive(samples=draws, return_inferencedata=True)
        
        print(f" Successfully sampled {draws} prior predictive draws")
            
    except Exception as e:
        error_msg = f"Failed to sample from prior predictive distribution: {str(e)}"
        print(f" {error_msg}")
        raise RuntimeError(error_msg) from e
    
    # Validate the results
    _validate_prior_predictive_results(idata)
    
    # Plot if requested
    if plot:
        try:
            plot_kwargs = plot_kwargs or {}
            _plot_prior_predictive(idata, y_obs, draws, custom_plots, **plot_kwargs)
        except Exception as e:
            print(f" Plotting failed: {e}")
            print(" Prior predictive samples were generated successfully")
    
    print(f" Prior predictive check completed successfully!")
    print(f" Results stored in InferenceData object")
    
    return idata

def _validate_model_structure(model: pm.Model):
    """Validate basic model structure before running prior predictive."""
    # Check if model has variables - use different approaches for PyMC compatibility
    try:
        # Try the standard PyMC approach
        if hasattr(model, 'vars') and model.vars:
            vars_list = list(model.vars)
        elif hasattr(model, 'variables') and model.variables:
            vars_list = list(model.variables)
        elif hasattr(model, 'free_RVs') and model.free_RVs:
            vars_list = list(model.free_RVs)
        else:
            # Fallback: try to get variables from the model context
            with model:
                vars_list = list(pm.Model.get_context().variables)
    except Exception:
        # If all else fails, assume the model is valid
        return
    
    if not vars_list:
        raise ValueError("Model has no variables defined")

def _infer_model_name(model: pm.Model) -> str:
    """Infer model name from PyMC model context."""
    # Try to get model name from context
    try:
        return model.name
    except AttributeError:
        pass
    
    # Fallback to default name
    return "model"

def _validate_prior_predictive_results(idata: az.InferenceData):
    """Validate that prior predictive results are reasonable."""
    # Check if we have prior predictive data
    if not hasattr(idata, 'prior_predictive') or idata.prior_predictive is None:
        raise ValueError("No prior predictive data found in InferenceData")
    
    # Check if we have any variables
    if not idata.prior_predictive.data_vars:
        raise ValueError("Prior predictive data has no variables")

def _get_response_variable_name(idata: az.InferenceData) -> str:
    """
    Get the response variable name from prior predictive data.
    
    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object with prior predictive samples
        
    Returns
    -------
    str
        Name of the response variable
    """
    available_vars = list(idata.prior_predictive.data_vars.keys())
    if 'y' in available_vars:
        return 'y'
    
    # Check for variables ending with '::y'
    y_vars = [v for v in available_vars if v.endswith('::y')]
    if y_vars:
        return y_vars[0]
    
    # Check for variables containing 'y' anywhere
    y_vars = [v for v in available_vars if 'y' in v]
    if y_vars:
        return y_vars[0]
    
    # Fallback: look for any variable that's not a parameter
    param_names = ['cutpoints', 'beta', 'sigma', 'mu', 'tau', 'cuts', 'alpha']
    response_vars = [v for v in available_vars if not any(param in v.lower() for param in param_names)]
    
    if response_vars:
        return response_vars[0]
    
    raise ValueError(f"Could not identify response variable in prior predictive data. Available variables: {available_vars}")

def _plot_prior_predictive(idata: az.InferenceData, y_obs: np.ndarray = None, draws: int = 200, custom_plots: dict = None, **kwargs):
    """
    Plot prior predictive distributions.
    
    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object with prior predictive samples
    y_obs : np.ndarray, optional
        Observed response variable for comparison
    draws : int
        Number of draws used for prior predictive sampling
    custom_plots : dict, optional
        Dictionary specifying which plots to show. If None, all plots are shown.
    **kwargs : dict
        Additional keyword arguments for plotting customization
    """
    # Get response variable name
    try:
        y_var_name = _get_response_variable_name(idata)
        y_data = idata.prior_predictive[y_var_name]
    except ValueError as e:
        print(f" {e}")
        return
    
    # Get number of categories from the data
    if y_obs is not None:
        unique_vals = np.unique(y_obs)
        K = len(unique_vals)
    else:
        unique_vals = np.unique(y_data.values)
        K = len(unique_vals)
    
    # Validate that data looks like ordinal categories
    if not np.allclose(y_data.values.flatten(), y_data.values.flatten().astype(int)):
        print(" Warning: Prior predictive data should be integer categories")
        return
    
    # Create plots
    try:
        # Determine which plots to show
        if custom_plots is None:
            # Default: show only mean distribution and observed (if available)
            show_plots = {
                'prior_samples': False,
                'mean_distribution': True,
                'observed': True,
                'category_counts': False,
                'total_observations': False,
                'category_proportions': False
            }
        else:
            # Use custom plot selection
            show_plots = {
                'prior_samples': custom_plots.get('prior_samples', False),
                'mean_distribution': custom_plots.get('mean_distribution', True),
                'observed': custom_plots.get('observed', True),
                'category_counts': custom_plots.get('category_counts', False),
                'total_observations': custom_plots.get('total_observations', False),
                'category_proportions': custom_plots.get('category_proportions', False)
            }
        
        # Count how many plots we'll show to determine layout
        n_plots = sum(show_plots.values())
        if n_plots == 0:
            print("  No plots requested, skipping visualization")
            return
        
        # Determine subplot layout based on number of plots
        if n_plots <= 3:
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
        elif n_plots <= 6:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
        else:
            # For more than 6 plots, use a grid
            cols = min(3, n_plots)
            rows = (n_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot 1: Prior predictive distribution across samples
        if show_plots['prior_samples']:
            plt.sca(axes[plot_idx])
            _plot_sample_distributions(y_data, K, draws)
            plot_idx += 1
        
        # Plot 2: Mean prior predictive distribution
        if show_plots['mean_distribution']:
            plt.sca(axes[plot_idx])
            _plot_mean_distribution(y_data, K)
            plot_idx += 1
        
        # Plot 3: Observed data (if available)
        if show_plots['observed'] and y_obs is not None:
            plt.sca(axes[plot_idx])
            _plot_observed_data(y_obs, K)
            plot_idx += 1
        
        # Plot 4: Distribution of category counts per sample
        if show_plots['category_counts']:
            plt.sca(axes[plot_idx])
            _plot_category_counts_distribution(y_data, K)
            plot_idx += 1
        
        # Plot 5: Distribution of total observations per sample
        if show_plots['total_observations']:
            plt.sca(axes[plot_idx])
            _plot_total_observations_distribution(y_data)
            plot_idx += 1
        
        # Plot 6: Category proportions across samples
        if show_plots['category_proportions']:
            plt.sca(axes[plot_idx])
            _plot_category_proportions(y_data, K)
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        print(" Prior predictive plots generated successfully")
            
    except Exception as e:
        print(f" Plotting failed: {e}")
        print("  Prior predictive data was generated successfully")
        raise RuntimeError(f"Failed to generate plots: {e}") from e

def _plot_sample_distributions(y_data, K, draws):
    """Plot individual sample distributions to show prior uncertainty."""
    # Show first 20 samples for clarity
    n_samples = min(20, draws)
    
    for i in range(n_samples):
        try:
            # Access data correctly based on dimensions
            if len(y_data.shape) == 3:  # (chain, draw, observations)
                sample_data = y_data.isel(chain=0, draw=i).values.flatten()
            elif len(y_data.shape) == 2 and "draw" in y_data.dims:  # (draw, observations)
                sample_data = y_data.isel(draw=i).values.flatten()
            else:  # 1D observations
                sample_data = y_data.values.flatten()
            
            # Only plot if we have valid data
            if len(sample_data) > 0 and not np.all(sample_data == 0):
                plt.hist(sample_data, bins=range(K+1), alpha=0.3, edgecolor='blue', 
                        color='lightblue', histtype='stepfilled', linewidth=1)
        except Exception as e:
            print(f"  Warning: Could not plot sample {i}: {e}")
            continue
    
    plt.xlabel("Ordinal Category")
    plt.ylabel("Frequency")
    plt.title("Prior Predictive Samples")
    plt.grid(True, alpha=0.3)

def _plot_mean_distribution(y_data, K):
    """Plot the mean frequency distribution across all samples."""
    try:
        # Compute mean frequency distribution across all draws
        if len(y_data.shape) == 3:  # (chain, draw, observations)
            # For each draw, count frequencies of each category
            freq_distributions = []
            for draw_idx in range(y_data.shape[1]):
                sample_data = y_data.isel(chain=0, draw=draw_idx).values.flatten()
                # Count frequencies for each category
                freq_counts = [np.sum(sample_data == cat) for cat in range(K)]
                freq_distributions.append(freq_counts)
            
            # Convert to numpy array and compute mean across draws
            freq_array = np.array(freq_distributions)  # Shape: (draws, categories)
            mean_freq = np.mean(freq_array, axis=0)   # Shape: (categories,)
            
        elif len(y_data.shape) == 2 and "draw" in y_data.dims:  # (draw, observations)
            freq_distributions = []
            for draw_idx in range(y_data.shape[0]):
                sample_data = y_data.isel(draw=draw_idx).values.flatten()
                freq_counts = [np.sum(sample_data == cat) for cat in range(K)]
                freq_distributions.append(freq_counts)
            
            freq_array = np.array(freq_distributions)
            mean_freq = np.mean(freq_array, axis=0)
            
        else:  # 1D observations
            sample_data = y_data.values.flatten()
            mean_freq = [np.sum(sample_data == cat) for cat in range(K)]
        
        # Plot the mean frequency distribution
        if len(mean_freq) > 0 and not np.all(mean_freq == 0):
            plt.bar(range(K), mean_freq, alpha=0.7, edgecolor='black', 
                   color='skyblue', linewidth=2, label="Mean Frequency Distribution")
        else:
            plt.text(0.5, 0.5, 'No valid mean data', ha='center', va='center', transform=plt.gca().transAxes)
            print(" Warning: Mean distribution has no valid data")
            
    except Exception as e:
        plt.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=plt.gca().transAxes)
        print(f" Warning: Could not compute mean distribution: {e}")
    
    plt.xlabel("Ordinal Category")
    plt.ylabel("Mean Frequency")
    plt.title("Mean Prior Predictive (Frequency Distribution)")
    plt.legend()
    plt.grid(True, alpha=0.3)

def _plot_observed_data(y_obs, K):
    """Plot observed data using bars."""
    # Count frequencies for each category
    freq_counts = [np.sum(y_obs == cat) for cat in range(K)]
    
    # Plot as bars instead of histogram
    plt.bar(range(K), freq_counts, alpha=0.7, edgecolor='black',
            color='lightcoral', label="Observed")
    plt.xlabel("Ordinal Category")
    plt.ylabel("Frequency")
    plt.title("Observed Data")
    plt.legend()
    plt.grid(True, alpha=0.3)



def _plot_category_counts_distribution(y_data, K):
    """Plot distribution of category counts per sample."""
    # Initialize counts for each category
    all_cat_counts = {cat: [] for cat in range(K)}
    
    # Use the correct dimension for draws
    max_draws = min(100, y_data.shape[1] if len(y_data.shape) > 1 else 1)
    
    for draw_idx in range(max_draws):
        try:
            # Access data correctly based on dimensions
            if len(y_data.shape) == 3:  # (chain, draw, observations)
                sample_data = y_data.isel(chain=0, draw=draw_idx).values.flatten()
            elif len(y_data.shape) == 2 and "draw" in y_data.dims:  # (draw, observations)
                sample_data = y_data.isel(draw=draw_idx).values.flatten()
            else:  # 1D observations
                sample_data = y_data.values.flatten()
            
            # Only process if we have valid data
            if len(sample_data) > 0 and not np.all(sample_data == 0):
                # Count occurrences of each category in this sample
                for cat in range(K):
                    cat_count = np.sum(sample_data == cat)
                    all_cat_counts[cat].append(cat_count)
                    
        except Exception:
            continue
    
    # Check if we have any data
    has_data = any(len(counts) > 0 for counts in all_cat_counts.values())
    
    if has_data:
        for cat in range(K):
            if all_cat_counts[cat]:  # Only plot if we have data for this category
                plt.hist(all_cat_counts[cat], alpha=0.7, label=f'Category {cat}', bins=20)
        
        plt.xlabel("Count per Sample")
        plt.ylabel("Frequency")
        plt.title("Category Counts per Sample")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=plt.gca().transAxes)

def _plot_total_observations_distribution(y_data):
    """Plot distribution of total observations per sample."""
    total_counts = []
    
    # Use the correct dimension for draws
    max_draws = min(100, y_data.shape[1] if len(y_data.shape) > 1 else 1)
    
    for draw_idx in range(max_draws):
        try:
            # Access data correctly based on dimensions
            if len(y_data.shape) == 3:  # (chain, draw, observations)
                sample_data = y_data.isel(chain=0, draw=draw_idx).values.flatten()
            elif len(y_data.shape) == 2 and "draw" in y_data.dims:  # (draw, observations)
                sample_data = y_data.isel(draw=draw_idx).values.flatten()
            else:  # 1D observations
                sample_data = y_data.values.flatten()
            
            # Only process if we have valid data
            if len(sample_data) > 0:
                total_count = len(sample_data)
                total_counts.append(total_count)
        except Exception:
            continue
    
    if total_counts:
        plt.hist(total_counts, alpha=0.7, color='green', bins=20)
        plt.xlabel("Total Observations per Sample")
        plt.ylabel("Frequency")
        plt.title("Total Observations per Sample")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=plt.gca().transAxes)

def _plot_category_proportions(y_data, K):
    """Plot category proportions across samples."""
    proportions = []
    
    # Use the correct dimension for draws
    max_draws = min(100, y_data.shape[1] if len(y_data.shape) > 1 else 1)
    
    for draw_idx in range(max_draws):
        try:
            # Access data correctly based on dimensions
            if len(y_data.shape) == 3:  # (chain, draw, observations)
                sample_data = y_data.isel(chain=0, draw=draw_idx).values.flatten()
            elif len(y_data.shape) == 2 and "draw" in y_data.dims:  # (draw, observations)
                sample_data = y_data.isel(draw=draw_idx).values.flatten()
            else:  # 1D observations
                sample_data = y_data.values.flatten()
            
            # Only process if we have valid data
            if len(sample_data) > 0 and not np.all(sample_data == 0):
                sample_props = []
                for cat in range(K):
                    prop = np.mean(sample_data == cat)
                    sample_props.append(prop)
                proportions.append(sample_props)
        except Exception:
            continue
    
    if proportions:
        proportions = np.array(proportions)
        # Box plot of proportions
        plt.boxplot([proportions[:, i] for i in range(K)], labels=[f'Cat {i}' for i in range(K)])
        plt.xlabel("Category")
        plt.ylabel("Proportion")
        plt.title("Category Proportions Across Samples")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=plt.gca().transAxes)
