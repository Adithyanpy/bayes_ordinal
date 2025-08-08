#!/usr/bin/env python3
"""
Trolley Problem Analysis - Probit Model Only

This script demonstrates a complete Bayesian workflow for ordinal regression
using the trolley problem dataset with PROBIT link function only.

The analysis includes:
- Data preparation and validation
- Model building (PyMC documentation style)
- Prior predictive checks
- Model fitting
- Diagnostics
- Posterior predictive checks
- Counterfactual analysis
- Visualization

References:
- PyMC Documentation: https://docs.pymc.io/en/latest/learn/core_notebooks/ordinal_regression.html
- McElreath, R. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and Stan.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pymc as pm
import sys
sys.path.append('..')
import bayes_ordinal as bo
from typing import Dict, Any, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

def load_and_clean_data(filepath: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, list]:
    """
    Load and clean the trolley problem dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the trolley dataset CSV file
        
    Returns
    -------
    tuple
        (cleaned_data, X, y, feature_names)
    """
    print("1. Loading and Cleaning Data...")
    print("=" * 50)
    
    # Load the dataset
    data = pd.read_csv(filepath, sep=';')
    print(f"✓ Raw dataset shape: {data.shape}")
    
    # Clean the data
    data = data.dropna()
    print(f"✓ Cleaned dataset shape: {data.shape}")
    
    # Check response variable
    print(f"✓ Response range: {data['response'].min()} to {data['response'].max()}")
    print(f"✓ Number of categories: {data['response'].nunique()}")
    print(f"✓ Features: {['action', 'intention', 'contact']}")
    
    # Display response distribution
    print("\nResponse distribution:")
    print(data['response'].value_counts().sort_index())
    
    # Prepare features
    feature_cols = ['action', 'intention', 'contact']
    X = data[feature_cols].values
    y = data['response'].values
    
    # Convert to 0-based indexing for PyMC
    y_zero_based = y - 1
    
    return data, X, y_zero_based, feature_cols

def validate_data(y_zero_based: np.ndarray, X: np.ndarray) -> None:
    """
    Validate the ordinal data using our package.
    
    Parameters
    ----------
    y_zero_based : np.ndarray
        Response variable (0-based)
    X : np.ndarray
        Feature matrix
    """
    print("\n2. Data Validation...")
    print("=" * 50)
    
    # Validate using our package
    validation_results = bo.validate_ordinal_data(y_zero_based, X)
    print("✓ Data validation completed")

def explore_data(data: pd.DataFrame, X: np.ndarray, feature_cols: list) -> None:
    """
    Explore the dataset characteristics.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset
    X : np.ndarray
        Feature matrix
    feature_cols : list
        Feature column names
    """
    print("\n3. Data Exploration...")
    print("=" * 50)
    
    # Feature summary
    print("Feature Summary:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: mean={X[:, i].mean():.3f}, std={X[:, i].std():.3f}")
    
    # Correlation matrix
    corr_matrix = pd.DataFrame(X, columns=feature_cols).corr()
    print(f"\nFeature Correlation Matrix:")
    print(corr_matrix.round(3))
    
    print("✓ Data exploration completed")

def get_priors() -> Dict[str, Any]:
    """
    Get default priors for the probit model.
        
    Returns
    -------
    dict
        Prior specifications
    """
    print("\n4. Prior Specification...")
    print("=" * 50)
    
    # Get default priors using our package
    priors = bo.get_default_priors()
    
    print("Prior Specifications:")
    for key, value in priors.items():
        print(f"  {key}: {value}")
    print("✓ Priors specified using our package")
    
    return priors

def build_model(y_zero_based: np.ndarray, X: np.ndarray, feature_cols: list, priors: Dict[str, Any]) -> pm.Model:
    """
    Build the probit model using our package.
    
    Parameters
    ----------
    y_zero_based : np.ndarray
        Response variable (0-based)
    X : np.ndarray
        Feature matrix
    feature_cols : list
        Feature column names
    priors : dict
        Prior specifications
        
    Returns
    -------
    pm.Model
        The probit model
    """
    print("\n5. Building Probit Model (Using Our Package)...")
    print("=" * 50)
    
    print("Building probit model...")
    
    # Build the probit model using our package
    probit_model = bo.cumulative_model(
        y=y_zero_based,
        X=X,
        link="probit",  # Use probit link
        priors=priors,
        model_name="trolley_probit",
        feature_names=feature_cols
    )
    
    print("✓ Probit model built successfully")
    
    # Display model structure
    print("\nModel Structure:")
    print("Model Structure (PyMC Graphviz):")
    
    # Create the graphviz visualization
    model_graph = pm.model_to_graphviz(probit_model)
    model_graph.render("trolley_probit_model", format="png", cleanup=True)
    print("✓ Model structure saved as trolley_probit_model.png")
    
    return probit_model

def run_prior_predictive_checks(probit_model: pm.Model, y_zero_based: np.ndarray) -> Any:
    """
    Run prior predictive checks for the probit model.
    
    Parameters
    ----------
    probit_model : pm.Model
        The probit model
    y_zero_based : np.ndarray
        Response variable (0-based)
        
    Returns
    -------
    Any
        Prior predictive samples
    """
    print("\n6. Prior Predictive Checks (Using Our Package)...")
    print("=" * 50)
    
    print("Running prior predictive checks for probit model...")
    
    # Run prior predictive checks
    prior_pred_probit = bo.run_prior_predictive(
        model=probit_model,
        draws=1000,
        plot=True,
        include_r2=True,
        y_obs=y_zero_based
    )
    
    print("✓ Probit prior predictive completed")
    
    return prior_pred_probit

def fit_model(probit_model: pm.Model) -> Any:
    """
    Fit the probit model using MCMC.
    
    Parameters
    ----------
    probit_model : pm.Model
        The probit model
        
    Returns
    -------
    Any
        Inference data
    """
    print("\n7. Model Fitting...")
    print("=" * 50)
    
    print("Fitting probit model...")
    
    # Fit the model using our package
    idata_probit = bo.fit_model(
        model=probit_model,
        chains=4,
        draws=1500,
        tune=1000
    )
    
    print("✓ Probit model fitting completed")
    
    return idata_probit

def run_diagnostics(idata_probit: Any) -> Dict:
    """
    Run diagnostics for the probit model.
    
    Parameters
    ----------
    idata_probit : Any
        Inference data for probit model
        
    Returns
    -------
    dict
        Diagnostic results
    """
    print("\n8. Model Diagnostics (Using Our Package)...")
    print("=" * 50)
    
    # Run comprehensive diagnostics
    results = bo.run_comprehensive_diagnostics(
        idata_probit,
        var_names=["trolley_probit::beta_contact", "trolley_probit::beta_intention", 
                  "trolley_probit::beta_action", "trolley_probit::alpha"],
        model_name="probit_model",
        include_plots=True,
        include_summary=True
    )
    
    print(f"✓ Model diagnostics completed")
    print(f"  Converged: {results['converged']}")
    if results["recommendations"]:
        print(f"  Recommendations: {', '.join(results['recommendations'])}")
    
    return results

def run_posterior_predictive_checks(probit_model: pm.Model, idata_probit: Any) -> Any:
    """
    Run posterior predictive checks for the probit model.
    
    Parameters
    ----------
    probit_model : pm.Model
        The probit model
    idata_probit : Any
        Inference data for probit model
        
    Returns
    -------
    Any
        Posterior predictive samples
    """
    print("\n9. Posterior Predictive Checks (Using Our Package)...")
    print("=" * 50)
    
    print("Running posterior predictive checks for probit model...")
    
    # Run posterior predictive checks
    ppc_probit = bo.run_posterior_predictive(
        probit_model,
        idata_probit,
        var_name="trolley_probit::response",
        kind="proportions"
    )
    
    print("✓ Probit posterior predictive completed")
    
    return ppc_probit

def visualize_parameters(idata_probit: Any, feature_cols: list) -> None:
    """
    Visualize probit model parameters.
    
    Parameters
    ----------
    idata_probit : Any
        Inference data for probit model
    feature_cols : list
        Feature column names
    """
    print("\n10. Parameter Visualization (Using ArviZ - Like OC Documentation)...")
    print("=" * 50)
    
    print("Plotting probit model parameters...")
    
    # Use ArviZ summary like OC documentation
    import arviz as az
    
    # Get summary of all parameters (like OC documentation)
    summary = az.summary(
        idata_probit,
        var_names=["trolley_probit::beta_contact", "trolley_probit::beta_intention", 
                  "trolley_probit::beta_action", "trolley_probit::alpha"]
    )
    
    print("Parameter Summary (Like OC Documentation):")
    print(summary)
    
    # Plot forest plot using ArviZ (like OC documentation)
    az.plot_forest(
        idata_probit,
        var_names=["trolley_probit::beta_contact", "trolley_probit::beta_intention", 
                  "trolley_probit::beta_action"],
        combined=True,
        figsize=(10, 6)
    )
    plt.title("Probit Coefficient Effects (Like OC Documentation)")
    plt.show()
    
    print("✓ Parameter visualization completed")

def run_counterfactual_analysis(probit_model: pm.Model, idata_probit: Any, feature_cols: list) -> Dict[str, Any]:
    """
    Run counterfactual analysis for the probit model.
    
    Parameters
    ----------
    probit_model : pm.Model
        The probit model
    idata_probit : Any
        Inference data for probit model
    feature_cols : list
        Feature column names
        
    Returns
    -------
    dict
        Counterfactual results
    """
    print("\n11. Counterfactual Analysis (Using Our Package)...")
    print("=" * 50)
    
    # Define counterfactual scenarios
    scenarios = {
        "Baseline": {"action": 0, "intention": 0, "contact": 0},
        "High Action": {"action": 1, "intention": 0, "contact": 0},
        "High Intention": {"action": 0, "intention": 1, "contact": 0},
        "High Contact": {"action": 0, "intention": 0, "contact": 1},
        "Action + Intention": {"action": 1, "intention": 1, "contact": 0},
        "Action + Contact": {"action": 1, "intention": 0, "contact": 1},
        "Intention + Contact": {"action": 0, "intention": 1, "contact": 1},
        "All High": {"action": 1, "intention": 1, "contact": 1}
    }
    
    print("Counterfactual scenarios:")
    for name, values in scenarios.items():
        print(f"  {name}: action={values['action']}, intention={values['intention']}, contact={values['contact']}")
    
    # Run counterfactual analysis using our fixed package
    counterfactual_results = bo.run_counterfactual_analysis(
        probit_model,
        idata_probit,
        scenarios,
        feature_names=feature_cols
    )
    
    print("✓ Counterfactual analysis completed")
    
    # Display results
    print("\nCounterfactual Analysis Results:")
    for scenario_name, result in counterfactual_results.items():
        print(f"  {scenario_name}: Mean = {result['mean']:.2f}, Std = {result['std']:.2f}")
    
    # Visualize results
    print("\nCreating counterfactual plots...")
    bo.plot_counterfactual_results(counterfactual_results)
    
    print("✓ Counterfactual visualization completed")
    
    return counterfactual_results

def create_results_summary(data: pd.DataFrame, feature_cols: list, 
                         conv_check_probit: Dict) -> None:
    """
    Create a summary of the probit analysis results.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset
    feature_cols : list
        Feature column names
    conv_check_probit : dict
        Convergence check results for probit model
    """
    print("\n12. Results Summary...")
    print("=" * 50)
    
    print("PROBIT MODEL ANALYSIS RESULTS")
    print("=" * 30)
    
    # Model convergence
    print(f"✓ Model converged: {conv_check_probit['converged']}")
    if 'summary' in conv_check_probit:
        summary = conv_check_probit['summary']
        if 'r_hat' in summary.columns:
            max_rhat = summary['r_hat'].max()
            print(f"✓ Max R-hat: {max_rhat:.3f}")
        if 'ess_bulk' in summary.columns:
            min_ess = summary['ess_bulk'].min()
            print(f"✓ Min ESS: {min_ess:.0f}")
    
    # Data summary
    print(f"\nData Summary:")
    print(f"✓ Observations: {len(data)}")
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Categories: {len(np.unique(data['response']))}")
    
    # Feature effects summary
    print(f"\nFeature Effects (Probit):")
    for name in feature_cols:
        var_name = f"trolley_probit::beta_{name}"
        try:
            if 'summary' in conv_check_probit:
                summary = conv_check_probit['summary']
                if var_name in summary.index:
                    mean_effect = summary.loc[var_name, 'mean']
                    hdi_low = summary.loc[var_name, 'hdi_3%']
                    hdi_high = summary.loc[var_name, 'hdi_97%']
                    print(f"  {name}: {mean_effect:.3f} [{hdi_low:.3f}, {hdi_high:.3f}]")
        except:
            print(f"  {name}: Effect not available")
    
    print("\n✓ Analysis completed successfully!")

def main():
    """
    Main function to run the complete probit analysis.
    """
    print("TROLLEY PROBLEM ANALYSIS - PROBIT MODEL ONLY")
    print("=" * 60)
    print("Complete Bayesian workflow for ordinal regression with probit link")
    print("=" * 60)
    
    # File path for the trolley dataset
    filepath = "../Dataset/Trolley.csv"
    
    try:
        # 1. Load and clean data
        data, X, y_zero_based, feature_cols = load_and_clean_data(filepath)
        
        # 2. Validate data
        validate_data(y_zero_based, X)
        
        # 3. Explore data
        explore_data(data, X, feature_cols)
        
        # 4. Get priors
        priors = get_priors()
        
        # 5. Build probit model
        probit_model = build_model(y_zero_based, X, feature_cols, priors)
        
        # 6. Run prior predictive checks
        prior_pred_probit = run_prior_predictive_checks(probit_model, y_zero_based)
        
        # 7. Fit probit model
        idata_probit = fit_model(probit_model)
        
        # 8. Run diagnostics
        conv_check_probit = run_diagnostics(idata_probit)
        
        # 9. Run posterior predictive checks
        ppc_probit = run_posterior_predictive_checks(probit_model, idata_probit)
        
        # 10. Visualize parameters
        visualize_parameters(idata_probit, feature_cols)
        
        # 11. Run counterfactual analysis
        counterfactual_results = run_counterfactual_analysis(probit_model, idata_probit, feature_cols)
        
        # 12. Create results summary
        create_results_summary(data, feature_cols, conv_check_probit)
        
        # Save results
        results = {
            "data": data,
            "feature_cols": feature_cols,
            "probit_model": probit_model,
            "idata_probit": idata_probit,
            "conv_check_probit": conv_check_probit,
            "prior_pred_probit": prior_pred_probit,
            "ppc_probit": ppc_probit,
            "counterfactual_results": counterfactual_results
        }
        
        import pickle
        with open("trolley_probit_analysis_results.pkl", "wb") as f:
            pickle.dump(results, f)
        print("\n✓ Results saved to trolley_probit_analysis_results.pkl")
        
        print("\n🎉 PROBIT ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 