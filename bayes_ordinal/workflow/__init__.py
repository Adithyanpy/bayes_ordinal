"""
Bayesian workflow components for ordinal regression.
"""

from .fitting import fit_model
from .diagnostics import summarize_diagnostics, plot_group_forest
from .prior_predictive import run_prior_predictive
from .posterior_predictive import run_posterior_predictive
from .cross_validation import compare_models, compare_models_stacking, compare_models_interpretation
from .sensitivity import prior_sensitivity, plot_influential
from .iterative import run_workflow
from .computation import (
    diagnose_computational_issues, simplify_model, run_on_subset,
    check_multimodality, reparameterize_model, add_prior_information,
    stack_individual_chains, fake_data_simulation, plot_intermediate_quantities,
    comprehensive_computation_check
)
from .model_modification import (
    pick_new_starting_model, replace_model_component, enrich_expand_model,
    use_approximation, add_more_data, modify_priors, compare_inferences,
    multiverse_analysis, model_averaging_stacking, systematic_model_modification,
    iterative_model_refinement
)

__all__ = [
    "fit_model",
    "summarize_diagnostics",
    "plot_group_forest",
    "run_prior_predictive",
    "run_posterior_predictive",
    "compare_models",
    "compare_models_stacking",
    "compare_models_interpretation",
    "prior_sensitivity",
    "plot_influential",
    "run_workflow",
    # Computational issue resolution
    "diagnose_computational_issues",
    "simplify_model",
    "run_on_subset",
    "check_multimodality",
    "reparameterize_model",
    "add_prior_information",
    "stack_individual_chains",
    "fake_data_simulation",
    "plot_intermediate_quantities",
    "comprehensive_computation_check",
    # Model modification
    "pick_new_starting_model",
    "replace_model_component",
    "enrich_expand_model",
    "use_approximation",
    "add_more_data",
    "modify_priors",
    "compare_inferences",
    "multiverse_analysis",
    "model_averaging_stacking",
    "systematic_model_modification",
    "iterative_model_refinement",
]
