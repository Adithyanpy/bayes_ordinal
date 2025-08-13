"""
Ordinal regression models for Bayesian inference.
"""

from .cumulative import cumulative_model
from .partial_odds import partial_odds_model

__all__ = [
    "cumulative_model",
    "partial_odds_model"
]
