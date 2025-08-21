import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Sequence

def summarize_diagnostics(
    idata: az.InferenceData, 
    var_names: list = None
) -> pd.DataFrame:
    """
    Summarize diagnostics with robust error handling.
    
    This function provides comprehensive diagnostic summaries including
    convergence metrics (R-hat, ESS), divergences, and energy diagnostics.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from model sampling.
    var_names : list, optional
        Specific variables to summarize. If None, summarizes all variables.
        
    Returns
    -------
    pd.DataFrame
        Diagnostic summary with convergence metrics.
        
    Examples
    --------
    >>> summary = summarize_diagnostics(idata)
    >>> summary = summarize_diagnostics(idata, var_names=["beta", "cutpoints"])
    """
    # Check if we have enough samples for reliable diagnostics
    n_chains = idata.posterior.dims.get('chain', 0)
    n_draws = idata.posterior.dims.get('draw', 0)
    
    if n_chains < 2 or n_draws < 4:
        print(f"Warning: Insufficient samples for reliable diagnostics (chains={n_chains}, draws={n_draws})")
        print("ArviZ requires at least 2 chains and 4 draws for accurate diagnostics")
        # Return basic summary without diagnostics that require multiple chains
        try:
            summary_df = az.summary(
                idata,
                var_names=var_names,
                round_to=2,
                hdi_prob=0.94
            )
        except Exception as e:
            print(f"Error computing basic summary: {e}")
            # Create minimal summary
            summary_df = pd.DataFrame({
                'mean': [0.0],
                'sd': [0.0],
                'hdi_3%': [0.0],
                'hdi_97%': [0.0],
                'r_hat': [1.0],
                'ess_bulk': [0.0],
                'ess_tail': [0.0]
            })
    else:
        # 1) Base summary: mean, sd, HDI, r_hat, ess_bulk, ess_tail
        try:
            summary_df = az.summary(
                idata,
                var_names=var_names,
                round_to=2,
                hdi_prob=0.94
            )
        except Exception as e:
            print(f"Error computing summary: {e}")
            # Create minimal summary
            summary_df = pd.DataFrame({
                'mean': [0.0],
                'sd': [0.0],
                'hdi_3%': [0.0],
                'hdi_97%': [0.0],
                'r_hat': [1.0],
                'ess_bulk': [0.0],
                'ess_tail': [0.0]
            })

    # 2) Count divergences per variable via sample_stats (if available)
    try:
        if "sample_stats" in idata and "diverging" in idata.sample_stats:
            divergences = idata.sample_stats["diverging"].sum(dim=["chain", "draw"]).item()
            summary_df["n_divergences"] = divergences
        else:
            summary_df["n_divergences"] = 0
    except Exception as e:
        print(f"Error computing divergences: {e}")
        summary_df["n_divergences"] = 0

    # 3) Max energy change from sample_stats (if available)
    try:
        if "sample_stats" in idata and "energy" in idata.sample_stats:
            energy_diff = idata.sample_stats["energy"].diff(dim="draw")
            max_energy = float(np.max(np.abs(energy_diff.values)))
            summary_df["max_energy_diff"] = max_energy
        else:
            summary_df["max_energy_diff"] = 0.0
    except Exception as e:
        print(f"Error computing energy diagnostics: {e}")
        summary_df["max_energy_diff"] = 0.0

    # 4) Check for log likelihood availability (needed for LOO/WAIC)
    try:
        if "log_likelihood" in idata:
            print(" Log likelihood available for LOO/WAIC calculations")
            summary_df["log_likelihood_available"] = True
        elif "log_likelihood" in idata.posterior_predictive:
            print(" Log likelihood available in posterior predictive for LOO/WAIC calculations")
            summary_df["log_likelihood_available"] = True
        elif "log_likelihood" in idata.sample_stats:
            print(" Log likelihood available in sample_stats for LOO/WAIC calculations")
            summary_df["log_likelihood_available"] = True
        else:
            print(" Log likelihood not found - LOO/WAIC calculations will fail")
            print("  This is needed for model comparison diagnostics")
            print("  Ensure you use log_likelihood=True in pm.sample()")
            summary_df["log_likelihood_available"] = False
    except Exception as e:
        print(f"Error checking log likelihood: {e}")
        summary_df["log_likelihood_available"] = False

    return summary_df

def run_comprehensive_diagnostics(
    idata: az.InferenceData,
    var_names: list = None,
    model_name: str = "model",
    include_plots: bool = True,
    include_summary: bool = True
) -> dict:
    """
    Run comprehensive diagnostics including summary and plots.
    
    This function provides a complete diagnostic assessment including
    convergence metrics, diagnostic plots, and recommendations.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from model sampling.
    var_names : list, optional
        Specific variables to analyze. If None, analyzes main parameters.
    model_name : str, default="model"
        Name of the model for the summary.
    include_plots : bool, default=True
        Whether to create diagnostic plots.
    include_summary : bool, default=True
        Whether to create diagnostic summary.
        
    Returns
    -------
    dict
        Comprehensive diagnostic results.
        
    Examples
    --------
    >>> results = run_comprehensive_diagnostics(idata, model_name="logit_model")
    """
    
    results = {
        "model_name": model_name,
        "summary": None,
        "recommendations": []
    }
    
    # 1. Get diagnostic summary
    if include_summary:
        try:
            summary_df = summarize_diagnostics(idata, var_names=var_names)
            results["summary"] = summary_df
            
            # Check convergence
            if "r_hat" in summary_df.columns:
                # Filter out NaN values and get max R-hat
                rhat_values = summary_df["r_hat"].dropna()
                if len(rhat_values) > 0:
                    max_rhat = rhat_values.max()
                    results["max_rhat"] = max_rhat
                    print(f"  Max R-hat: {max_rhat:.3f}")
                else:
                    results["max_rhat"] = None
                    print("  No valid R-hat values found")
            else:
                results["max_rhat"] = None
                print("  R-hat column not found in summary")
            
            if "ess_bulk" in summary_df.columns:
                min_ess = summary_df["ess_bulk"].min()
                results["min_ess"] = min_ess
                
                if min_ess < 400:
                    results["recommendations"].append("Low ESS - consider increasing draws")
            
            if "n_divergences" in summary_df.columns:
                n_divergences = summary_df["n_divergences"].sum()
                results["n_divergences"] = n_divergences
                
                if n_divergences > 0:
                    results["recommendations"].append(f"Found {n_divergences} divergences - check model specification")
            
            print(f" Diagnostic summary completed")
            print(f"  Model: {model_name}")
            print(f"  Summary shape: {summary_df.shape}")
            print(f"  R-hat column exists: {'r_hat' in summary_df.columns}")
            if 'r_hat' in summary_df.columns:
                print(f"  R-hat values: {summary_df['r_hat'].values}")
            if results["recommendations"]:
                print(f"  Recommendations: {', '.join(results['recommendations'])}")
                
        except Exception as e:
            print(f" Diagnostic summary failed: {e}")
    
    # 2. Create diagnostic plots
    if include_plots:
        try:
            plot_diagnostics(idata, var_names=var_names)
            print(f" Diagnostic plots completed")
        except Exception as e:
            print(f" Diagnostic plots failed: {e}")
    
    return results

def plot_diagnostics(
    idata: az.InferenceData,
    var_names: list = None,
    include_energy: bool = True,
    include_trace: bool = True,
    include_rank: bool = True,
    include_autocorr: bool = True
) -> None:
    """
    Plot comprehensive diagnostic plots for Bayesian model assessment.
    
    This function creates diagnostic plots including energy plots, trace plots,
    rank plots, and autocorrelation plots following PyMC best practices.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from model sampling.
    var_names : list, optional
        Specific variables to plot. If None, plots all variables.
    include_energy : bool, default=True
        Whether to include energy plots.
    include_trace : bool, default=True
        Whether to include trace plots.
    include_rank : bool, default=True
        Whether to include rank plots.
    include_autocorr : bool, default=True
        Whether to include autocorrelation plots.
    """
    
    if var_names is None:
        # Use main parameters if no specific variables provided
        # Look for common parameter patterns in the model (prioritizing PyMCOrdinal convention)
        available_vars = list(idata.posterior.data_vars.keys())
        var_names = []
        
        # Add beta parameters (coefficients) - PyMCOrdinal docs use "beta"
        beta_vars = [v for v in available_vars if "beta" in v.lower()]
        var_names.extend(beta_vars)
        
        # Add cutpoints parameters - PyMCOrdinal docs use "cutpoints" (not "alpha")
        cutpoint_vars = [v for v in available_vars if "cutpoints" in v.lower()]
        var_names.extend(cutpoint_vars)
        
        # If no specific patterns found, use first few variables
        if not var_names and available_vars:
            var_names = available_vars[:5]  # Limit to first 5 variables
    
    print("Creating diagnostic plots...")
    
    # 1. Energy plot (most important for convergence)
    if include_energy and "sample_stats" in idata and "energy" in idata.sample_stats:
        try:
            print(" Creating energy plot...")
            az.plot_energy(idata)
            plt.title("Energy Plot - Check for Divergences")
            plt.show()
        except Exception as e:
            print(f" Energy plot failed: {e}")
    
    # 2. Trace plots
    if include_trace:
        try:
            print(" Creating trace plots...")
            az.plot_trace(idata, var_names=var_names)
            plt.show()
        except Exception as e:
            print(f" Trace plots failed: {e}")
    
    # 3. Rank plots
    if include_rank:
        try:
            print(" Creating rank plots...")
            az.plot_rank(idata, var_names=var_names)
            plt.show()
        except Exception as e:
            print(f" Rank plots failed: {e}")
    
    # 4. Autocorrelation plots
    if include_autocorr:
        try:
            print(" Creating autocorrelation plots...")
            az.plot_autocorr(idata, var_names=var_names)
            plt.show()
        except Exception as e:
            print(f" Autocorrelation plots failed: {e}")
    

def plot_group_forest(
    idata: az.InferenceData,
    var_name: str,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] | None = None,
    combined: bool = True
):
    """
    Plot a forest (caterpillar) plot for group-level effects.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data from PyMC model.
    var_name : str
        Name of the group-level variable (e.g., "u").
    hdi_prob : float, default=0.94
        Width of the HDI for display.
    figsize : tuple, optional
        Figure size for the plot.
    combined : bool, default=True
        Whether to combine chains in the plot.

    Examples
    --------
    >>> plot_group_forest(idata, var_name="u", hdi_prob=0.9)
    """
    az.plot_forest(
        idata,
        var_names=[var_name],
        hdi_prob=hdi_prob,
        combined=combined,
        figsize=figsize,
        kind="forestplot"
    )
    plt.title(f"Group‐level `{var_name}` — {int(hdi_prob*100)}% CI")
    plt.show()

def create_model_summary(
    idata: az.InferenceData, 
    model_name: str = "model"
) -> dict:
    """
    Create a comprehensive model summary following PyMC best practices.
    
    This function provides a complete model assessment including sampling
    information, diagnostics, parameter summaries, and recommendations.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from model sampling.
    model_name : str, default="model"
        Name of the model for the summary.
        
    Returns
    -------
    dict
        Comprehensive model summary including diagnostics and recommendations.
        
    Examples
    --------
    >>> summary = create_model_summary(idata, model_name="logit_model")
    """
    summary = {
        "model_name": model_name,
        "sampling_info": {},
        "diagnostics": {},
        "parameter_summary": {},
        "recommendations": []
    }
    
    # Sampling information
    summary["sampling_info"] = {
        "n_chains": idata.posterior.dims.get("chain", 0),
        "n_draws": idata.posterior.dims.get("draw", 0),
        "n_parameters": len(idata.posterior.data_vars),
        "sampling_time": getattr(idata, "sample_stats", {}).get("sampling_time", None)
    }
    
    # Diagnostics
    diag_df = summarize_diagnostics(idata)
    summary["diagnostics"] = {
        "max_rhat": diag_df["r_hat"].max() if "r_hat" in diag_df else None,
        "min_ess": diag_df["ess_bulk"].min() if "ess_bulk" in diag_df else None,
        "n_divergences": diag_df["n_divergences"].sum() if "n_divergences" in diag_df else 0,
    }
    
    # Parameter summary
    if hasattr(idata, "posterior"):
        for var_name in idata.posterior.data_vars:
            var_data = idata.posterior[var_name].values.flatten()
            summary["parameter_summary"][var_name] = {
                "mean": float(np.mean(var_data)),
                "std": float(np.std(var_data)),
                "q2_5": float(np.percentile(var_data, 2.5)),
                "q97_5": float(np.percentile(var_data, 97.5))
            }
    
    # Recommendations based on diagnostics
    if summary["diagnostics"]["max_rhat"] and summary["diagnostics"]["max_rhat"] > 1.01:
        summary["recommendations"].append("Increase number of tuning steps or draws")
    
    if summary["diagnostics"]["min_ess"] and summary["diagnostics"]["min_ess"] < 400:
        summary["recommendations"].append("Increase number of draws for better ESS")
    
    if summary["diagnostics"]["n_divergences"] > 0:
        summary["recommendations"].append("Check for model specification issues or use different initialization")
    
    return summary
