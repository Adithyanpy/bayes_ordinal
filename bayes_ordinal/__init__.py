"""
Bayesian Ordinal Regression Package

A PyMC-based package for Bayesian ordinal regression with comprehensive workflow tools.
"""

# Core models
from .models.cumulative import cumulative_model
from .models.partial_odds import partial_odds_model

# Analysis functions
from .analysis.counterfactual import run_counterfactual_analysis, plot_counterfactual_results

# Utility functions
from .utils.model_inspection import inspect_model_variables, print_model_summary

# Workflow components
from .workflow.fitting import fit_ordinal_model
from .workflow.diagnostics import summarize_diagnostics, plot_diagnostics, plot_group_forest, create_model_summary, run_comprehensive_diagnostics
from .workflow.prior_predictive import run_prior_predictive
from .workflow.posterior_predictive import run_posterior_predictive

from .workflow.cross_validation import compare_models, compare_models_stacking, compare_models_interpretation, plot_model_comparison_interpretation, display_comparison_results
from .workflow.sensitivity import prior_sensitivity, plot_influential

from .workflow.computation import (
    diagnose_computational_issues, run_on_subset,
    check_multimodality, stack_individual_chains, fake_data_simulation, 
    comprehensive_computation_check
)

# Utilities
from .plot_utils import set_plot_saving, save_current_plot, show_and_save, reset_save_counter
from .utils import (
    validate_ordinal_data, encode_categorical_features, standardize_features,
    create_group_indices, compute_category_proportions, check_convergence,
    validate_ordinal_model
)
from .plotting import (
    plot_model_comparison, plot_prior_posterior, plot_forest,
    plot_contrast, draw_causal_graph, plot_model_comparison_interpretation,
    plot_model_structure, plot_cutpoints, plot_coefficient_effects,
    plot_category_probabilities, plot_prior_predictive, plot_posterior_predictive
)

# Configuration system
from .config import (
    Config, ModelConfig, PriorConfig, SamplingConfig, WorkflowConfig, 
    DataConfig, OutputConfig, get_default_config, get_quick_test_config, 
    get_production_config, get_hierarchical_config
)

# Version
__version__ = "0.1.0"

__all__ = [
    # Models
    "cumulative_model",
    "run_counterfactual_analysis",
    "plot_counterfactual_results",
    "partial_odds_model",

    # Workflow
    "fit_ordinal_model",
    "summarize_diagnostics",
    "plot_diagnostics",
    "plot_group_forest",
    "create_model_summary",
    "run_comprehensive_diagnostics",
    "run_prior_predictive",
    "run_posterior_predictive",

    "compare_models",
    "compare_models_stacking",
    "compare_models_interpretation",
    "plot_model_comparison_interpretation",
    "display_comparison_results",
    "prior_sensitivity",
    "plot_influential",

    # Computational issue resolution
    "diagnose_computational_issues",
    "run_on_subset",
    "check_multimodality",
    "stack_individual_chains",
    "fake_data_simulation",

    "comprehensive_computation_check",
    # Utilities
    "set_plot_saving",
    "save_current_plot", 
    "show_and_save",
    "reset_save_counter",
    "validate_ordinal_data",
    "encode_categorical_features",
    "standardize_features",
    "create_group_indices",
    "compute_category_proportions",
    "check_convergence",
    "validate_ordinal_model",

    # Plotting (standalone functions)
    "plot_model_comparison",
    "plot_prior_posterior",
    "plot_forest",
    "plot_contrast",
    "draw_causal_graph",
    "plot_model_structure",
    "plot_cutpoints",
    "plot_coefficient_effects",
    "plot_category_probabilities",
    # Plotting (delegations to workflow functions)
    "plot_prior_predictive",  # Delegates to run_prior_predictive()
    "plot_posterior_predictive",  # Delegates to run_posterior_predictive()
    "plot_model_comparison_interpretation",  # Delegates to workflow function

    # Configuration
    "Config",
    "ModelConfig",
    "PriorConfig",
    "SamplingConfig",
    "WorkflowConfig",
    "DataConfig",
    "OutputConfig",
    "get_default_config",
    "get_quick_test_config",
    "get_production_config",
    "get_hierarchical_config",
]
