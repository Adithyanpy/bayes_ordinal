"""
Bayesian workflow components for ordinal regression.
"""

from .fitting import fit_ordinal_model
from .diagnostics import summarize_diagnostics, plot_group_forest
from .prior_predictive import run_prior_predictive
from .posterior_predictive import run_posterior_predictive
from .cross_validation import compare_models, compare_models_stacking, compare_models_interpretation, display_comparison_results
from .sensitivity import prior_sensitivity, plot_influential

from .computation import (
    diagnose_computational_issues,
    check_multimodality, stack_individual_chains, fake_data_simulation, 
    comprehensive_computation_check
)

__all__ = [
    "fit_ordinal_model",
    "summarize_diagnostics",
    "plot_group_forest",
    "run_prior_predictive",
    "run_posterior_predictive",
    "compare_models",
    "compare_models_stacking",
    "compare_models_interpretation",
    "display_comparison_results",
    "prior_sensitivity",
    "plot_influential",

    # Computational issue resolution
    "diagnose_computational_issues",
    "check_multimodality",
    "stack_individual_chains",
    "fake_data_simulation",

    "comprehensive_computation_check",
]
