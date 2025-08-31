"""
Bayesian Ordinal Regression Package

A PyMC-based package for Bayesian ordinal regression with comprehensive workflow tools.
"""

# Core models
from .models.cumulative import cumulative_model

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
    diagnose_computational_issues,
    check_multimodality, stack_individual_chains, fake_data_simulation, 
    comprehensive_computation_check
)

# Utilities
from .utils.data_processing import (
    validate_ordinal_data, encode_categorical_features, standardize_features,
    create_group_indices, compute_category_proportions
)
from .utils.model_validation import (
    validate_ordinal_model
)




# Version
__version__ = "0.1.0"

__all__ = [
    # Models
    "cumulative_model",
    "run_counterfactual_analysis",
    "plot_counterfactual_results",

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
    "check_multimodality",
    "stack_individual_chains",
    "fake_data_simulation",

    "comprehensive_computation_check",
    # Utilities

    "validate_ordinal_data",
    "encode_categorical_features",
    "standardize_features",
    "create_group_indices",
    "compute_category_proportions",
    "validate_ordinal_model",




]
