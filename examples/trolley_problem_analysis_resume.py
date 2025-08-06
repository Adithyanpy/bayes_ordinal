#!/usr/bin/env python3
"""
Trolley Problem Analysis - Complete Bayesian Workflow (Resumable Version)

This script demonstrates a complete Bayesian workflow for ordinal regression
using the trolley problem dataset, following PyMC documentation structure.

The analysis compares logit and probit link functions and includes:
- Data preparation and validation
- Model building (PyMC documentation style)
- Prior predictive checks
- Model fitting
- Diagnostics
- Posterior predictive checks
- Model comparison
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
import bayes_ordinal as bo
from typing import Dict, Any, Tuple
import warnings
import pickle
import os

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
    data['response'] = data['response'].astype(int)
    data = data.dropna()
    data = data[data['response'].between(1, 7)]
    
    # Define feature columns
    feature_cols = ['action', 'intention', 'contact']
    feature_names = ['Action', 'Intention', 'Contact']
    
    # Create feature matrix and response vector
    X = data[feature_cols].values
    y = data['response'].values
    
    print(f"✓ Cleaned dataset shape: {data.shape}")
    print(f"✓ Response range: {y.min()} to {y.max()}")
    print(f"✓ Number of categories: {len(np.unique(y))}")
    print(f"✓ Features: {feature_names}")
    
    # Display response distribution
    print("\nResponse distribution:")
    response_counts = pd.Series(y).value_counts().sort_index()
    print(response_counts)
    
    return data, X, y, feature_names

def validate_data(y: np.ndarray, X: np.ndarray, K: int) -> None:
    """
    Validate the ordinal data using our package.
    
    Parameters
    ----------
    y : np.ndarray
        Response variable
    X : np.ndarray
        Feature matrix
    K : int
        Number of categories
    """
    print("\n2. Data Validation...")
    print("=" * 50)
    
    # Use our package's validation function
    validation_result = bo.validate_ordinal_model(y, X, K)
    
    for warning in validation_result['warnings']:
        print(f"⚠ {warning}")
        
    for recommendation in validation_result['recommendations']:
        print(f"💡 {recommendation}")
    
    print("✓ Data validation completed")

def explore_data(data: pd.DataFrame, X: np.ndarray, feature_names: list) -> None:
    """
    Explore and visualize the data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Cleaned dataset
    X : np.ndarray
        Feature matrix
    feature_names : list
        Names of features
    """
    print("\n3. Data Exploration...")
    print("=" * 50)
    
    # Feature summary
    print("Feature Summary:")
    for i, name in enumerate(feature_names):
        print(f"  {name}: mean={X[:, i].mean():.3f}, std={X[:, i].std():.3f}")
    
    # Correlation matrix
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr()
    print(f"\nFeature Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('trolley_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot response distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    response_counts = pd.Series(data['response']).value_counts().sort_index()
    plt.bar(response_counts.index, response_counts.values, alpha=0.7, color='skyblue')
    plt.xlabel('Response Category')
    plt.ylabel('Frequency')
    plt.title('Trolley Problem Response Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(data['response'], bins=range(1, 9), alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Response Category')
    plt.ylabel('Frequency')
    plt.title('Response Distribution (Histogram)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trolley_response_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Data exploration completed")

def get_priors(X: np.ndarray, K: int) -> Dict[str, Any]:
    """
    Get priors using our package's prior functions.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    K : int
        Number of categories
        
    Returns
    -------
    dict
        Prior specifications
    """
    print("\n4. Prior Specification...")
    print("=" * 50)
    
    # Use our package's conservative priors
    priors = bo.get_conservative_priors(K=K, model_type="ordinal")
    
    print("Prior Specifications:")
    for param, spec in priors.items():
        print(f"  {param}: {spec}")
    
    print("✓ Priors specified using our package")
    return priors

def build_models(data: pd.DataFrame, feature_cols: list, feature_names: list, priors: Dict[str, Any]) -> Tuple[pm.Model, pm.Model]:
    """
    Build logit and probit models using our package.
    
    Parameters
    ----------
    data : pd.DataFrame
        Cleaned dataset
    feature_cols : list
        Feature column names
    feature_names : list
        Feature display names
    priors : dict
        Prior specifications
        
    Returns
    -------
    tuple
        (logit_model, probit_model)
    """
    print("\n5. Building Models (Using Our Package)...")
    print("=" * 50)
    
    # Build logit model using our package
    print("Building logit model...")
    logit_model = bo.cumulative_model(
        y=data['response'],
        X=data[feature_cols],
        link="logit",
        priors=priors,
        model_name="trolley_logit"
    )
    print("✓ Logit model built successfully")
    
    # Build probit model using our package
    print("Building probit model...")
    probit_model = bo.cumulative_model(
        y=data['response'],
        X=data[feature_cols],
        link="probit",
        priors=priors,
        model_name="trolley_probit"
    )
    print("✓ Probit model built successfully")
    
    # Display model structures using our package
    print("\nModel Structures:")
    bo.plot_model_structure(logit_model)
    plt.savefig('trolley_logit_model_structure.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    bo.plot_model_structure(probit_model)
    plt.savefig('trolley_probit_model_structure.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return logit_model, probit_model

def run_prior_predictive_checks(logit_model: pm.Model, probit_model: pm.Model, 
                              data: pd.DataFrame, feature_cols: list) -> Tuple[Any, Any]:
    """
    Run prior predictive checks using our package.
    
    Parameters
    ----------
    logit_model : pm.Model
        Logit model
    probit_model : pm.Model
        Probit model
    data : pd.DataFrame
        Cleaned dataset
    feature_cols : list
        Feature column names
        
    Returns
    -------
    tuple
        (prior_idata_logit, prior_idata_probit)
    """
    print("\n6. Prior Predictive Checks (Using Our Package)...")
    print("=" * 50)
    
    # Use our package's prior predictive function
    print("Running prior predictive checks for logit model...")
    prior_idata_logit = bo.run_prior_predictive(
        logit_model,
        draws=200,
        plot=True,
        y_obs=data['response'].values
    )
    print("✓ Logit prior predictive completed")
    
    print("Running prior predictive checks for probit model...")
    prior_idata_probit = bo.run_prior_predictive(
        probit_model,
        draws=200,
        plot=True,
        y_obs=data['response'].values
    )
    print("✓ Probit prior predictive completed")
    
    return prior_idata_logit, prior_idata_probit

def fit_models(logit_model: pm.Model, probit_model: pm.Model, 
               data: pd.DataFrame, feature_cols: list) -> Tuple[Any, Any]:
    """
    Fit both models using our package's workflow.
    
    Parameters
    ----------
    logit_model : pm.Model
        Logit model
    probit_model : pm.Model
        Probit model
    data : pd.DataFrame
        Cleaned dataset
    feature_cols : list
        Feature column names
        
    Returns
    -------
    tuple
        (idata_logit, idata_probit)
    """
    print("\n7. Model Fitting...")
    print("=" * 50)
    
    # Fit logit model
    print("Fitting logit model...")
    with logit_model:
        pm.set_data({
            "action": data['action'].astype(float),
            "intention": data['intention'].astype(float),
            "contact": data['contact'].astype(float)
        })
        
        idata_logit = pm.sample(
            draws=1000,
            tune=500,
            chains=4,
            return_inferencedata=True,
            random_seed=42
        )
    
    print("✓ Logit model fitting completed")
    
    # Fit probit model
    print("Fitting probit model...")
    with probit_model:
        pm.set_data({
            "action": data['action'].astype(float),
            "intention": data['intention'].astype(float),
            "contact": data['contact'].astype(float)
        })
        
        idata_probit = pm.sample(
            draws=1000,
            tune=500,
            chains=4,
            return_inferencedata=True,
            random_seed=42
        )
    
    print("✓ Probit model fitting completed")
    
    return idata_logit, idata_probit

def run_diagnostics(idata_logit: Any, idata_probit: Any) -> Tuple[Dict, Dict]:
    """
    Run diagnostics using our package.
    
    Parameters
    ----------
    idata_logit : Any
        Inference data for logit model
    idata_probit : Any
        Inference data for probit model
        
    Returns
    -------
    tuple
        (conv_check_logit, conv_check_probit)
    """
    print("\n8. Model Diagnostics (Using Our Package)...")
    print("=" * 50)
    
    # Use our package's convergence check
    print("Checking convergence for logit model...")
    conv_check_logit = bo.check_convergence(idata_logit)
    print(f"✓ Logit converged: {conv_check_logit['converged']}")
    
    print("Checking convergence for probit model...")
    conv_check_probit = bo.check_convergence(idata_probit)
    print(f"✓ Probit converged: {conv_check_probit['converged']}")
    
    # Use our package's diagnostics summary
    diag_logit = bo.summarize_diagnostics(idata_logit)
    diag_probit = bo.summarize_diagnostics(idata_probit)
    
    print("\nLogit Model Diagnostics:")
    print(diag_logit[['r_hat', 'ess_bulk', 'ess_tail', 'n_divergences']].round(3))
    
    print("\nProbit Model Diagnostics:")
    print(diag_probit[['r_hat', 'ess_bulk', 'ess_tail', 'n_divergences']].round(3))
    
    # Skip advanced diagnostics plotting for stability
    print("\nSkipping advanced diagnostics plotting for stability...")
    print("✓ Basic convergence diagnostics completed successfully")
    
    return conv_check_logit, conv_check_probit

def run_posterior_predictive_checks(logit_model: pm.Model, probit_model: pm.Model,
                                  idata_logit: Any, idata_probit: Any,
                                  data: pd.DataFrame, feature_cols: list) -> Tuple[Any, Any]:
    """
    Run posterior predictive checks using our package.
    
    Parameters
    ----------
    logit_model : pm.Model
        Logit model
    probit_model : pm.Model
        Probit model
    idata_logit : Any
        Inference data for logit model
    idata_probit : Any
        Inference data for probit model
    data : pd.DataFrame
        Cleaned dataset
    feature_cols : list
        Feature column names
        
    Returns
    -------
    tuple
        (ppc_logit, ppc_probit)
    """
    print("\n9. Posterior Predictive Checks (Using Our Package)...")
    print("=" * 50)
    
    # Use our package's posterior predictive function
    print("Running posterior predictive checks for logit model...")
    ppc_logit = bo.run_posterior_predictive(
        logit_model,
        idata_logit,
        var_name="trolley_logit::response",
        kind="proportions"
    )
    print("✓ Logit posterior predictive completed")
    
    print("Running posterior predictive checks for probit model...")
    ppc_probit = bo.run_posterior_predictive(
        probit_model,
        idata_probit,
        var_name="trolley_probit::response",
        kind="proportions"
    )
    print("✓ Probit posterior predictive completed")
    
    return ppc_logit, ppc_probit

def visualize_parameters(idata_logit: Any, idata_probit: Any, feature_names: list) -> None:
    """
    Visualize parameter estimates using our package.
    
    Parameters
    ----------
    idata_logit : Any
        Inference data for logit model
    idata_probit : Any
        Inference data for probit model
    feature_names : list
        Feature names
    """
    print("\n10. Parameter Visualization (Using Our Package)...")
    print("=" * 50)
    
    # Use our package's plotting functions
    print("Plotting logit model parameters...")
    
    # Plot individual beta coefficients for logit model
    beta_vars = ["trolley_logit::beta_action", "trolley_logit::beta_intention", "trolley_logit::beta_contact"]
    for i, var_name in enumerate(beta_vars):
        az.plot_forest(
            idata_logit, 
            var_names=[var_name], 
            combined=True,
            figsize=(8, 4)
        )
        plt.title(f'Logit Model: {feature_names[i]} Effect')
        plt.tight_layout()
        plt.savefig(f'trolley_logit_{feature_names[i].lower()}_effect.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    bo.plot_cutpoints(
        idata_logit, 
        var_name="trolley_logit::alpha"
    )
    plt.savefig('trolley_logit_cutpoints.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot coefficient effects for probit model
    print("Plotting probit model parameters...")
    
    # Plot individual beta coefficients for probit model
    beta_vars = ["trolley_probit::beta_action", "trolley_probit::beta_intention", "trolley_probit::beta_contact"]
    for i, var_name in enumerate(beta_vars):
        az.plot_forest(
            idata_probit, 
            var_names=[var_name], 
            combined=True,
            figsize=(8, 4)
        )
        plt.title(f'Probit Model: {feature_names[i]} Effect')
        plt.tight_layout()
        plt.savefig(f'trolley_probit_{feature_names[i].lower()}_effect.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    bo.plot_cutpoints(
        idata_probit, 
        var_name="trolley_probit::alpha"
    )
    plt.savefig('trolley_probit_cutpoints.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_models(logit_model: pm.Model, probit_model: pm.Model,
                  idata_logit: Any, idata_probit: Any) -> Dict[str, Any]:
    """
    Compare logit and probit models using our package.
    
    Parameters
    ----------
    logit_model : pm.Model
        Logit model
    probit_model : pm.Model
        Probit model
    idata_logit : Any
        Inference data for logit model
    idata_probit : Any
        Inference data for probit model
        
    Returns
    -------
    dict
        Model comparison results
    """
    print("\n11. Model Comparison (Using Our Package)...")
    print("=" * 50)
    
    # Prepare models and inference data for comparison
    models = {
        "logit": logit_model,
        "probit": probit_model
    }
    
    idatas = {
        "logit": idata_logit,
        "probit": idata_probit
    }
    
    # Use our package's model comparison functions
    print("Comparing models using LOO...")
    comparison_loo = bo.compare_models(models, idatas, ic="loo")
    print("\nLOO Model Comparison:")
    print(comparison_loo.round(3))
    
    print("\nComparing models using WAIC...")
    comparison_waic = bo.compare_models(models, idatas, ic="waic")
    print("\nWAIC Model Comparison:")
    print(comparison_waic.round(3))
    
    # Use our package's comprehensive model comparison
    print("\nRunning comprehensive model comparison...")
    comparison_results = bo.compare_models_interpretation(models, idatas, ic="loo")
    
    print(f"\nBest Model: {comparison_results['best_model']}")
    print(f"Information Criterion: {comparison_results['ic_values'][0]:.2f}")
    
    print("\nModel Interpretations:")
    for name, interp in comparison_results['interpretation'].items():
        print(f"  {name}: {interp['status']} - {interp['description']}")
    
    # Use our package's model comparison plotting
    bo.plot_model_comparison_interpretation(comparison_results)
    plt.savefig('trolley_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'loo': comparison_loo,
        'waic': comparison_waic,
        'comprehensive': comparison_results
    }

def run_counterfactual_analysis(logit_model: pm.Model, idata_logit: Any,
                               feature_cols: list, feature_names: list) -> Dict[str, Any]:
    """
    Run counterfactual analysis using our package.
    
    Parameters
    ----------
    logit_model : pm.Model
        Logit model (assumed to be the best model)
    idata_logit : Any
        Inference data for logit model
    feature_cols : list
        Feature column names
    feature_names : list
        Feature display names
        
    Returns
    -------
    dict
        Counterfactual analysis results
    """
    print("\n12. Counterfactual Analysis (Using Our Package)...")
    print("=" * 50)
    
    # Create counterfactual scenarios
    scenarios = {
        'Baseline': np.array([0, 0, 0]),
        'High Action': np.array([1, 0, 0]),
        'High Intention': np.array([0, 1, 0]),
        'High Contact': np.array([0, 0, 1]),
        'Action + Intention': np.array([1, 1, 0]),
        'Action + Contact': np.array([1, 0, 1]),
        'Intention + Contact': np.array([0, 1, 1]),
        'All High': np.array([1, 1, 1])
    }
    
    print("Counterfactual scenarios:")
    for name, values in scenarios.items():
        print(f"  {name}: Action={values[0]}, Intention={values[1]}, Contact={values[2]}")
    
    # Use our package's counterfactual analysis
    counterfactual_results = bo.run_counterfactual_analysis(
        model=logit_model,
        idata=idata_logit,
        scenarios=scenarios,
        feature_names=["action", "intention", "contact"],  # Use actual variable names
        n_samples=1000
    )
    
    print("✓ Counterfactual analysis completed")
    
    # Use our package's counterfactual plotting
    bo.plot_counterfactual_results(counterfactual_results)
    plt.savefig('trolley_counterfactual_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print counterfactual summary
    print("\nCounterfactual Analysis Summary:")
    for scenario_name, result in counterfactual_results.items():
        print(f"  {scenario_name}:")
        print(f"    Mean: {result['mean']:.2f}")
        print(f"    Std: {result['std']:.2f}")
        print(f"    95% CI: [{result['percentiles'][0]:.1f}, {result['percentiles'][4]:.1f}]")
    
    return counterfactual_results

def create_results_summary(data: pd.DataFrame, feature_names: list, 
                          conv_check_logit: Dict, conv_check_probit: Dict,
                          comparison_results: Dict[str, Any]) -> None:
    """
    Create a comprehensive results summary.
    
    Parameters
    ----------
    data : pd.DataFrame
        Cleaned dataset
    feature_names : list
        Feature names
    conv_check_logit : dict
        Convergence check for logit model
    conv_check_probit : dict
        Convergence check for probit model
    comparison_results : dict
        Model comparison results
    """
    print("\n13. Results Summary...")
    print("=" * 60)
    
    print(f"📊 Dataset Information:")
    print(f"   • Sample size: {len(data)}")
    print(f"   • Number of categories: {len(np.unique(data['response']))}")
    print(f"   • Features: {', '.join(feature_names)}")
    print(f"   • Response range: {data['response'].min()}-{data['response'].max()}")
    
    print(f"\n🔬 Model Performance:")
    loo_results = comparison_results['loo']
    print(f"   • Logit LOO: {loo_results.loc['logit', 'elpd_loo']:.2f}")
    print(f"   • Probit LOO: {loo_results.loc['probit', 'elpd_loo']:.2f}")
    print(f"   • Best model: {comparison_results['comprehensive']['best_model']}")
    
    print(f"\n✅ Convergence:")
    print(f"   • Logit: {conv_check_logit['converged']}")
    print(f"   • Probit: {conv_check_probit['converged']}")
    
    print(f"\n📈 Key Findings:")
    for name, interp in comparison_results['comprehensive']['interpretation'].items():
        print(f"   • {name}: {interp['description']}")
    
    print(f"\n🎯 Recommendations:")
    for rec in comparison_results['comprehensive']['recommendations']['model_selection'][:3]:
        print(f"   • {rec}")
    
    print(f"\n📝 Notes:")
    print(f"   • Response categories: 1-{len(np.unique(data['response']))}")
    print(f"   • Analysis follows PyMC Ordered Categories documentation")
    print(f"   • Models use pm.Data() for flexible predictions")
    print(f"   • Counterfactual analysis available for 'what-if' scenarios")
    print(f"   • All functions use our bayes_ordinal package")
    
    print("\n✓ Analysis completed successfully!")

def save_checkpoint(results: Dict[str, Any], step: str) -> None:
    """
    Save checkpoint data to resume later.
    
    Parameters
    ----------
    results : dict
        Results to save
    step : str
        Current step name
    """
    checkpoint_file = f'trolley_checkpoint_{step}.pkl'
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Checkpoint saved: {checkpoint_file}")

def load_checkpoint(step: str) -> Dict[str, Any]:
    """
    Load checkpoint data to resume from.
    
    Parameters
    ----------
    step : str
        Step name to load from
        
    Returns
    -------
    dict
        Loaded results
    """
    checkpoint_file = f'trolley_checkpoint_{step}.pkl'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            results = pickle.load(f)
        print(f"✓ Checkpoint loaded: {checkpoint_file}")
        return results
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_file}")
        return {}

def main(resume_from: str = None):
    """
    Main function to run the complete trolley problem analysis.
    
    Parameters
    ----------
    resume_from : str, optional
        Step to resume from. Options: 'data_loaded', 'models_built', 'models_fitted', 'diagnostics_complete'
    """
    print("TROLLEY PROBLEM ANALYSIS - COMPLETE BAYESIAN WORKFLOW")
    print("=" * 60)
    print("Using Our bayes_ordinal Package")
    print("Following PyMC Documentation Structure")
    print("Comparing Logit vs Probit Link Functions")
    print("=" * 60)
    
    # File path for the trolley dataset
    filepath = '../Dataset/Trolley.csv'
    
    # Initialize results dictionary
    results = {}
    
    try:
        # Check if we should resume from a checkpoint
        if resume_from and resume_from != 'start':
            results = load_checkpoint(resume_from)
            print(f"Resuming from step: {resume_from}")
        
        # 1. Load and clean data
        if resume_from is None or resume_from == 'start':
            data, X, y, feature_names = load_and_clean_data(filepath)
            feature_cols = ['action', 'intention', 'contact']
            
            results.update({
                'data': data,
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'feature_cols': feature_cols
            })
            save_checkpoint(results, 'data_loaded')
        else:
            data = results['data']
            X = results['X']
            y = results['y']
            feature_names = results['feature_names']
            feature_cols = results['feature_cols']
            print("✓ Data loaded from checkpoint")
        
        # 2. Validate data using our package
        if resume_from is None or resume_from == 'start':
            validate_data(y, X, len(np.unique(y)))
        
        # 3. Explore data
        if resume_from is None or resume_from == 'start':
            explore_data(data, X, feature_names)
        
        # 4. Get priors using our package
        if resume_from is None or resume_from == 'start':
            priors = get_priors(X, len(np.unique(y)))
            results['priors'] = priors
        
        # 5. Build models using our package
        if resume_from is None or resume_from in ['start', 'data_loaded']:
            logit_model, probit_model = build_models(data, feature_cols, feature_names, results.get('priors', None))
            results.update({
                'models': {'logit': logit_model, 'probit': probit_model}
            })
            save_checkpoint(results, 'models_built')
        else:
            logit_model = results['models']['logit']
            probit_model = results['models']['probit']
            print("✓ Models loaded from checkpoint")
        
        # 6. Prior predictive checks using our package
        if resume_from is None or resume_from in ['start', 'data_loaded']:
            prior_idata_logit, prior_idata_probit = run_prior_predictive_checks(
                logit_model, probit_model, data, feature_cols
            )
            results.update({
                'prior_idatas': {'logit': prior_idata_logit, 'probit': prior_idata_probit}
            })
        
        # 7. Fit models
        if resume_from is None or resume_from in ['start', 'data_loaded', 'models_built']:
            idata_logit, idata_probit = fit_models(logit_model, probit_model, data, feature_cols)
            results.update({
                'idatas': {'logit': idata_logit, 'probit': idata_probit}
            })
            save_checkpoint(results, 'models_fitted')
        else:
            idata_logit = results['idatas']['logit']
            idata_probit = results['idatas']['probit']
            print("✓ Model fits loaded from checkpoint")
        
        # 8. Run diagnostics using our package
        if resume_from is None or resume_from in ['start', 'data_loaded', 'models_built']:
            conv_check_logit, conv_check_probit = run_diagnostics(idata_logit, idata_probit)
            results.update({
                'conv_check_logit': conv_check_logit,
                'conv_check_probit': conv_check_probit
            })
            save_checkpoint(results, 'diagnostics_complete')
        else:
            conv_check_logit = results['conv_check_logit']
            conv_check_probit = results['conv_check_probit']
            print("✓ Diagnostics loaded from checkpoint")
        
        # 9. Posterior predictive checks using our package
        if resume_from is None or resume_from in ['start', 'data_loaded', 'models_built']:
            ppc_logit, ppc_probit = run_posterior_predictive_checks(
                logit_model, probit_model, idata_logit, idata_probit, data, feature_cols
            )
            results.update({
                'ppc': {'logit': ppc_logit, 'probit': ppc_probit}
            })
        
        # 10. Visualize parameters using our package
        visualize_parameters(idata_logit, idata_probit, feature_names)
        
        # 11. Compare models using our package
        comparison_results = compare_models(logit_model, probit_model, idata_logit, idata_probit)
        results['comparison_results'] = comparison_results
        
        # 12. Counterfactual analysis using our package
        counterfactual_results = run_counterfactual_analysis(
            logit_model, idata_logit, feature_cols, feature_names
        )
        results['counterfactual_results'] = counterfactual_results
        
        # 13. Create results summary
        create_results_summary(data, feature_names, conv_check_logit, conv_check_probit, comparison_results)
        
        # Save results
        print("\n14. Saving Results...")
        print("=" * 50)
        
        with open('trolley_analysis_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print("✓ Results saved to 'trolley_analysis_results.pkl'")
        print("✓ All plots saved as PNG files")
        print("\n🎉 Analysis completed successfully using our bayes_ordinal package!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # Check if resume argument is provided
    resume_from = None
    if len(sys.argv) > 1:
        resume_from = sys.argv[1]
        print(f"Resuming from: {resume_from}")
    
    main(resume_from) 