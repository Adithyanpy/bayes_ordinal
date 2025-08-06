"""
Bayesian Ordinal Regression Package

A PyMC-based package for Bayesian ordinal regression with comprehensive workflow tools.
"""

# Core models
from .models.cumulative import cumulative_model, run_counterfactual_analysis, plot_counterfactual_results
from .models.partial_odds import partial_odds_model
from .models.pymcordinal_compatible import (
    constrained_uniform, make_model, sample_model, create_pymcordinal_workflow
)

# Workflow components
from .workflow.fitting import fit_model
from .workflow.diagnostics import summarize_diagnostics, plot_diagnostics, plot_group_forest, create_model_summary, run_comprehensive_diagnostics
from .workflow.prior_predictive import run_prior_predictive
from .workflow.posterior_predictive import run_posterior_predictive
from .workflow.cross_validation import compare_models, compare_models_stacking, compare_models_interpretation, plot_model_comparison_interpretation
from .workflow.sensitivity import prior_sensitivity, plot_influential
from .workflow.iterative import run_workflow
from .workflow.computation import (
    diagnose_computational_issues, simplify_model, run_on_subset,
    check_multimodality, reparameterize_model, add_prior_information,
    stack_individual_chains, fake_data_simulation, plot_intermediate_quantities,
    comprehensive_computation_check
)
from .workflow.model_modification import (
    pick_new_starting_model, replace_model_component, enrich_expand_model,
    use_approximation, add_more_data, modify_priors, compare_inferences,
    multiverse_analysis, model_averaging_stacking, systematic_model_modification,
    iterative_model_refinement
)

# Utilities
from .priors import (
    get_default_priors, get_weakly_informative_priors, get_informative_priors, 
    get_conservative_priors, get_data_informed_priors, validate_priors
)
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
    # PyMCOrdinal-compatible functions
    "constrained_uniform",
    "make_model",
    "sample_model",
    "create_pymcordinal_workflow",

    # Workflow
    "fit_model",
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

    # Utilities
    "get_default_priors",
    "get_weakly_informative_priors",
    "get_informative_priors",
    "get_conservative_priors",
    "get_data_informed_priors",
    "validate_priors",
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
