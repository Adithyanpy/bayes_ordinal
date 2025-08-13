"""
Analysis functions for Bayesian ordinal models.

This module contains functions for analyzing fitted models, including
counterfactual analysis, diagnostics, and model comparison.
"""

from .counterfactual import (
    run_counterfactual_analysis,
    plot_counterfactual_results
)

__all__ = [
    "run_counterfactual_analysis",
    "plot_counterfactual_results"
]
