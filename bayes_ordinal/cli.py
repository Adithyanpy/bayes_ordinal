#!/usr/bin/env python3
"""
Command-line interface for Bayesian ordinal regression.

This module provides a CLI for running Bayesian ordinal regression analyses
from the command line with configuration files.
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import bayes_ordinal as bo
from .config import Config, get_default_config, get_quick_test_config, get_production_config


def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load data from various file formats.
    
    Parameters
    ----------
    filepath : str
        Path to data file (CSV, Excel, etc.)
        
    Returns
    -------
    y : np.ndarray
        Response variable
    X : np.ndarray
        Feature matrix
    K : int
        Number of categories
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.csv':
        data = pd.read_csv(filepath)
    elif filepath.suffix.lower() in ['.xlsx', '.xls']:
        data = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Assume last column is response, rest are features
    y = data.iloc[:, -1].values
    X = data.iloc[:, :-1].values
    
    # Determine number of categories
    K = len(np.unique(y))
    
    return y, X, K


def save_results(results: Dict[str, Any], output_dir: str, config: Config) -> None:
    """
    Save analysis results to files.
    
    Parameters
    ----------
    results : dict
        Analysis results
    output_dir : str
        Output directory
    config : Config
        Configuration object
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save(output_path / "config.yaml")
    
    # Save results summary
    summary = {
        'models_built': list(results.get('models', {}).keys()),
        'inference_data': list(results.get('idata', {}).keys()),
        'diagnostics': list(results.get('diagnostics', {}).keys()),
        'cross_validation': type(results.get('cv')).__name__ if 'cv' in results else None
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save cross-validation results
    if 'cv' in results and hasattr(results['cv'], 'to_csv'):
        results['cv'].to_csv(output_path / "model_comparison.csv")
    
    # Save diagnostics
    for name, diag in results.get('diagnostics', {}).items():
        if hasattr(diag, 'to_csv'):
            diag.to_csv(output_path / f"diagnostics_{name}.csv")
    
    print(f"Results saved to: {output_path}")


def run_analysis(config_path: Optional[str], data_path: str, output_dir: str, 
                quick_test: bool = False, production: bool = False) -> None:
    """
    Run Bayesian ordinal regression analysis.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file
    data_path : str
        Path to data file
    output_dir : str
        Output directory
    quick_test : bool
        Use quick test configuration
    production : bool
        Use production configuration
    """
    print("=" * 60)
    print("BAYESIAN ORDINAL REGRESSION ANALYSIS")
    print("=" * 60)
    
    # Load configuration
    if quick_test:
        config = get_quick_test_config()
        print("Using quick test configuration")
    elif production:
        config = get_production_config()
        print("Using production configuration")
    elif config_path:
        config = Config.from_file(config_path)
        print(f"Loaded configuration from: {config_path}")
    else:
        config = get_default_config()
        print("Using default configuration")
    
    # Load data
    print(f"Loading data from: {data_path}")
    y, X, K = load_data(data_path)
    print(f"Data loaded: {len(y)} samples, {X.shape[1]} features, {K} categories")
    
    # Data preprocessing
    if config.data.validate_data:
        print("Validating data...")
        y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K)
    
    if config.data.standardize_features:
        print("Standardizing features...")
        X_scaled, scaler = bo.standardize_features(X_clean)
    else:
        X_scaled = X_clean
    
    # Get priors based on configuration
    if config.priors.prior_type == "default":
        priors = bo.get_default_priors()
    elif config.priors.prior_type == "weakly_informative":
        priors = bo.get_weakly_informative_priors()
    elif config.priors.prior_type == "informative":
        priors = bo.get_informative_priors(
            coef_means=config.priors.coef_mu,
            coef_sds=config.priors.coef_sigma,
            cut_means=config.priors.cut_mu,
            cut_sds=config.priors.cut_sigma
        )
    else:
        priors = config.priors.custom_priors
    
    # Set random seed
    if config.sampling.random_seed is not None:
        np.random.seed(config.sampling.random_seed)
    
    # Define models based on configuration
    models = {}
    if config.model.model_type == "cumulative":
        models["cumulative"] = bo.cumulative_model
    elif config.model.model_type == "partial_odds":
        models["partial_odds"] = bo.partial_odds_model
    else:
        # Run both models for comparison
        models = {
            "cumulative": bo.cumulative_model,
            "partial_odds": bo.partial_odds_model
        }
    
    print(f"Building {len(models)} model(s)...")
    
    # Run workflow
    workflow_config = config.get_workflow_config()
    
    results = bo.run_workflow(
        model_fn=models,
        y=y_clean, X=X_scaled, K=K,
        priors=priors,
        config=workflow_config
    )
    
    # Save results
    if config.output.save_results:
        save_results(results, output_dir, config)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if 'cv' in results and hasattr(results['cv'], 'index'):
        print("Model Comparison Results:")
        print(results['cv'])
        
        best_model = results['cv'].index[0]
        print(f"\nBest model: {best_model}")
        print(f"LOO difference: {results['cv'].iloc[0]['elpd_diff']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")


def create_config_template(output_path: str) -> None:
    """
    Create a configuration template file.
    
    Parameters
    ----------
    output_path : str
        Path to save the template
    """
    config = get_default_config()
    config.name = "my_ordinal_analysis"
    config.description = "Template configuration for Bayesian ordinal regression"
    config.save(output_path)
    print(f"Configuration template saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bayesian Ordinal Regression CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  bayes-ordinal data.csv -o results/
  
  # Run with custom configuration
  bayes-ordinal data.csv -c config.yaml -o results/
  
  # Quick test run
  bayes-ordinal data.csv --quick-test -o results/
  
  # Production run
  bayes-ordinal data.csv --production -o results/
  
  # Create configuration template
  bayes-ordinal --create-config config_template.yaml
        """
    )
    
    # Main arguments
    parser.add_argument(
        'data_path',
        nargs='?',
        help='Path to data file (CSV, Excel)'
    )
    
    parser.add_argument(
        '-c', '--config',
        help='Path to configuration file (YAML/JSON)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='./results',
        help='Output directory (default: ./results)'
    )
    
    # Configuration presets
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Use quick test configuration (fast, minimal samples)'
    )
    
    parser.add_argument(
        '--production',
        action='store_true',
        help='Use production configuration (robust, many samples)'
    )
    
    # Utility commands
    parser.add_argument(
        '--create-config',
        metavar='FILE',
        help='Create a configuration template file'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'bayes-ordinal {bo.__version__}'
    )
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.create_config:
        create_config_template(args.create_config)
        return
    
    # Validate main arguments
    if not args.data_path:
        parser.error("Data file path is required")
    
    if not Path(args.data_path).exists():
        parser.error(f"Data file not found: {args.data_path}")
    
    if args.quick_test and args.production:
        parser.error("Cannot use both --quick-test and --production")
    
    if args.config and not Path(args.config).exists():
        parser.error(f"Configuration file not found: {args.config}")
    
    # Run analysis
    try:
        run_analysis(
            config_path=args.config,
            data_path=args.data_path,
            output_dir=args.output,
            quick_test=args.quick_test,
            production=args.production
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 