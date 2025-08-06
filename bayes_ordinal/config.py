"""
Configuration system for Bayesian ordinal regression.

This module provides a comprehensive configuration system for managing
model parameters, priors, and workflow settings.
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import numpy as np
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model specification."""
    
    # Model type
    model_type: str = "cumulative"  # "cumulative" or "partial_odds"
    
    # Link function for cumulative models
    link: str = "logit"  # "logit", "probit", "cloglog"
    
    # Hierarchical structure
    hierarchical: bool = False
    group_variable: Optional[str] = None
    
    # Model parameters
    n_categories: Optional[int] = None
    n_features: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_type not in ["cumulative", "partial_odds"]:
            raise ValueError(f"model_type must be 'cumulative' or 'partial_odds', got {self.model_type}")
        
        if self.model_type == "cumulative" and self.link not in ["logit", "probit", "cloglog"]:
            raise ValueError(f"link must be 'logit', 'probit', or 'cloglog', got {self.link}")


@dataclass
class PriorConfig:
    """Configuration for prior specification."""
    
    # Prior type
    prior_type: str = "default"  # "default", "weakly_informative", "informative", "custom"
    
    # Coefficient priors
    coef_mu: Union[float, np.ndarray] = 0.0
    coef_sigma: Union[float, np.ndarray] = 2.5
    
    # Cutpoint priors (for cumulative models)
    cut_mu: Union[float, np.ndarray] = 0.0
    cut_sigma: Union[float, np.ndarray] = 5.0
    
    # Group-level priors (for hierarchical models)
    u_sigma: float = 1.0
    
    # Gamma priors (for partial odds models)
    gamma_mu: Union[float, np.ndarray] = 0.0
    gamma_sigma: Union[float, np.ndarray] = 5.0
    
    # Custom prior specification
    custom_priors: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.prior_type not in ["default", "weakly_informative", "informative", "custom"]:
            raise ValueError(f"prior_type must be one of ['default', 'weakly_informative', 'informative', 'custom'], got {self.prior_type}")


@dataclass
class SamplingConfig:
    """Configuration for MCMC sampling."""
    
    # Sampling parameters
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    
    # Initialization
    init: str = "jitter+adapt_diag"
    jitter_max_retries: int = 5
    
    # Smoke test
    smoke_test: bool = False
    
    # Progress bar
    progressbar: bool = True
    
    # Random seed
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.chains < 1:
            raise ValueError(f"chains must be >= 1, got {self.chains}")
        if self.draws < 1:
            raise ValueError(f"draws must be >= 1, got {self.draws}")
        if self.tune < 0:
            raise ValueError(f"tune must be >= 0, got {self.tune}")


@dataclass
class WorkflowConfig:
    """Configuration for workflow steps."""
    
    # Workflow steps
    run_prior_predictive: bool = True
    run_posterior_predictive: bool = True
    run_diagnostics: bool = True
    run_cross_validation: bool = True
    run_sensitivity_analysis: bool = False
    
    # Prior predictive
    prior_draws: int = 200
    
    # Posterior predictive
    ppc_kind: str = "proportions"  # "proportions", "hist", "ecdf"
    
    # Cross-validation
    cv_metric: str = "loo"  # "loo", "waic"
    cv_threshold: float = 0.7
    
    # Diagnostics
    hdi_prob: float = 0.94
    rhat_threshold: float = 1.1
    ess_threshold: float = 400
    
    # Sensitivity analysis
    sensitivity_hyperparameters: List[str] = field(default_factory=lambda: ["coef_sigma", "cut_sigma"])
    sensitivity_values: Dict[str, List[float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.cv_metric not in ["loo", "waic"]:
            raise ValueError(f"cv_metric must be 'loo' or 'waic', got {self.cv_metric}")
        if self.ppc_kind not in ["proportions", "hist", "ecdf"]:
            raise ValueError(f"ppc_kind must be one of ['proportions', 'hist', 'ecdf'], got {self.ppc_kind}")


@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    
    # Data validation
    validate_data: bool = True
    check_missing: bool = True
    check_categories: bool = True
    
    # Feature preprocessing
    standardize_features: bool = True
    encode_categorical: bool = True
    
    # Categorical columns (if known)
    categorical_columns: Optional[List[str]] = None
    
    # Group structure
    group_columns: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        pass


@dataclass
class OutputConfig:
    """Configuration for output and visualization."""
    
    # Output directory
    output_dir: Optional[str] = None
    
    # Save results
    save_results: bool = True
    save_format: str = "json"  # "json", "yaml", "pickle"
    
    # Plots
    save_plots: bool = True
    plot_format: str = "png"  # "png", "pdf", "svg"
    dpi: int = 300
    
    # Verbosity
    verbose: bool = True
    show_plots: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.save_format not in ["json", "yaml", "pickle"]:
            raise ValueError(f"save_format must be one of ['json', 'yaml', 'pickle'], got {self.save_format}")
        if self.plot_format not in ["png", "pdf", "svg"]:
            raise ValueError(f"plot_format must be one of ['png', 'pdf', 'svg'], got {self.plot_format}")


@dataclass
class Config:
    """Main configuration class that combines all sub-configurations."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    priors: PriorConfig = field(default_factory=PriorConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Metadata
    name: str = "bayes_ordinal_analysis"
    description: str = ""
    version: str = "0.1.0"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()
        
        # Update sub-configurations
        for key, value in config_dict.items():
            if hasattr(config, key) and isinstance(value, dict):
                sub_config = getattr(config, key)
                for sub_key, sub_value in value.items():
                    if hasattr(sub_config, sub_key):
                        setattr(sub_config, sub_key, sub_value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "Config":
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif filepath.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'model': {
                'model_type': self.model.model_type,
                'link': self.model.link,
                'hierarchical': self.model.hierarchical,
                'group_variable': self.model.group_variable,
                'n_categories': self.model.n_categories,
                'n_features': self.model.n_features
            },
            'priors': {
                'prior_type': self.priors.prior_type,
                'coef_mu': self.priors.coef_mu,
                'coef_sigma': self.priors.coef_sigma,
                'cut_mu': self.priors.cut_mu,
                'cut_sigma': self.priors.cut_sigma,
                'u_sigma': self.priors.u_sigma,
                'gamma_mu': self.priors.gamma_mu,
                'gamma_sigma': self.priors.gamma_sigma,
                'custom_priors': self.priors.custom_priors
            },
            'sampling': {
                'draws': self.sampling.draws,
                'tune': self.sampling.tune,
                'chains': self.sampling.chains,
                'init': self.sampling.init,
                'jitter_max_retries': self.sampling.jitter_max_retries,
                'smoke_test': self.sampling.smoke_test,
                'progressbar': self.sampling.progressbar,
                'random_seed': self.sampling.random_seed
            },
            'workflow': {
                'run_prior_predictive': self.workflow.run_prior_predictive,
                'run_posterior_predictive': self.workflow.run_posterior_predictive,
                'run_diagnostics': self.workflow.run_diagnostics,
                'run_cross_validation': self.workflow.run_cross_validation,
                'run_sensitivity_analysis': self.workflow.run_sensitivity_analysis,
                'prior_draws': self.workflow.prior_draws,
                'ppc_kind': self.workflow.ppc_kind,
                'cv_metric': self.workflow.cv_metric,
                'cv_threshold': self.workflow.cv_threshold,
                'hdi_prob': self.workflow.hdi_prob,
                'rhat_threshold': self.workflow.rhat_threshold,
                'ess_threshold': self.workflow.ess_threshold,
                'sensitivity_hyperparameters': self.workflow.sensitivity_hyperparameters,
                'sensitivity_values': self.workflow.sensitivity_values
            },
            'data': {
                'validate_data': self.data.validate_data,
                'check_missing': self.data.check_missing,
                'check_categories': self.data.check_categories,
                'standardize_features': self.data.standardize_features,
                'encode_categorical': self.data.encode_categorical,
                'categorical_columns': self.data.categorical_columns,
                'group_columns': self.data.group_columns
            },
            'output': {
                'output_dir': self.output.output_dir,
                'save_results': self.output.save_results,
                'save_format': self.output.save_format,
                'save_plots': self.output.save_plots,
                'plot_format': self.output.plot_format,
                'dpi': self.output.dpi,
                'verbose': self.output.verbose,
                'show_plots': self.output.show_plots
            }
        }
        
        return config_dict
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif filepath.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration as dictionary for compatibility."""
        return {
            'draws': self.sampling.draws,
            'tune': self.sampling.tune,
            'chains': self.sampling.chains,
            'smoke_test': self.sampling.smoke_test,
            'progressbar': self.sampling.progressbar,
            'run_prior': self.workflow.run_prior_predictive,
            'run_ppc': self.workflow.run_posterior_predictive,
            'run_diagnostics': self.workflow.run_diagnostics,
            'run_cv': self.workflow.run_cross_validation,
            'prior_draws': self.workflow.prior_draws,
            'ppc_kind': self.workflow.ppc_kind,
            'cv_metric': self.workflow.cv_metric
        }


# Predefined configurations
def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_quick_test_config() -> Config:
    """Get configuration for quick testing."""
    config = Config()
    config.sampling.draws = 100
    config.sampling.tune = 50
    config.sampling.chains = 2
    config.sampling.smoke_test = True
    config.workflow.run_sensitivity_analysis = False
    config.output.verbose = False
    config.output.show_plots = False
    return config


def get_production_config() -> Config:
    """Get configuration for production analysis."""
    config = Config()
    config.sampling.draws = 2000
    config.sampling.tune = 1000
    config.sampling.chains = 4
    config.sampling.smoke_test = False
    config.workflow.run_sensitivity_analysis = True
    config.output.save_results = True
    config.output.save_plots = True
    return config


def get_hierarchical_config() -> Config:
    """Get configuration for hierarchical models."""
    config = Config()
    config.model.hierarchical = True
    config.priors.prior_type = "weakly_informative"
    config.sampling.draws = 1500
    config.sampling.tune = 750
    return config 