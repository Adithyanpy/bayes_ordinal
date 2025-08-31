"""
Utility functions for Bayesian ordinal models.

This module contains helper functions for model inspection, diagnostics,
data processing, and other utilities that work across different model types.
"""

from .model_inspection import (
    inspect_model_variables,
    print_model_summary
)

from .data_processing import (
    validate_ordinal_data,
    encode_categorical_features,
    standardize_features,
    create_group_indices,
    compute_category_proportions
)

from .model_validation import (
    validate_ordinal_model
)

__all__ = [
    # Model inspection
    "inspect_model_variables",
    "print_model_summary",
    
    # Data processing
    "validate_ordinal_data",
    "encode_categorical_features",
    "standardize_features",
    "create_group_indices",
    "compute_category_proportions",
    
    # Model validation
    "validate_ordinal_model"
]
