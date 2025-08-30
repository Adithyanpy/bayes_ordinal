"""
Model inspection utilities for Bayesian ordinal models.

This module provides functions for inspecting and understanding the structure
of fitted PyMC models, useful for debugging and analysis.
"""

from typing import Dict, Any
import pymc as pm


def inspect_model_variables(model: pm.Model) -> Dict[str, Any]:
    """
    Inspect model variables to help understand the model structure.
    
    This is useful for debugging and understanding what variables
    are available for analysis.
    
    Parameters
    ----------
    model : pm.Model
        The PyMC model to inspect
        
    Returns
    -------
    dict
        Dictionary containing model variable information
    """
    info = {
        "free_variables": {},
        "observed_variables": {},
        "deterministic_variables": {},
        "data_variables": {},
        "model_name": getattr(model, 'name', 'unnamed_model')
    }
    
    # Free variables (parameters)
    for rv in model.free_RVs:
        info["free_variables"][rv.name] = {
            "type": str(type(rv)),
            "shape": getattr(rv, 'shape', 'unknown'),
            "dtype": getattr(rv, 'dtype', 'unknown')
        }
    
    # Observed variables
    for rv in model.observed_RVs:
        info["observed_variables"][rv.name] = {
            "type": str(type(rv)),
            "shape": getattr(rv, 'shape', 'unknown'),
            "dtype": getattr(rv, 'dtype', 'unknown')
        }
    
    # Deterministic variables
    for var in model.deterministics:
        info["deterministic_variables"][var.name] = {
            "type": str(type(var)),
            "shape": getattr(var, 'shape', 'unknown'),
            "dtype": getattr(var, 'dtype', 'unknown')
        }
    
    # Data variables (pm.Data containers)
    for name, var in model.named_vars.items():
        if hasattr(var, 'get_value'):
            info["data_variables"][name] = {
                "type": "pm.Data",
                "shape": getattr(var, 'shape', 'unknown'),
                "dtype": getattr(var, 'dtype', 'unknown')
            }
    
    return info


def print_model_summary(model: pm.Model) -> None:
    """
    Print a human-readable summary of the model structure.
    
    Parameters
    ----------
    model : pm.Model
        The PyMC model to summarize
    """
    info = inspect_model_variables(model)
    
    print(f"\n{'='*60}")
    print(f"Model: {info['model_name']}")
    print(f"{'='*60}")
    
    print(f"\n FREE VARIABLES (Parameters):")
    if info['free_variables']:
        for name, details in info['free_variables'].items():
            print(f"  - {name}: {details['type']}, shape: {details['shape']}")
    else:
        print("  None")
    
    print(f"\n OBSERVED VARIABLES:")
    if info['observed_variables']:
        for name, details in info['observed_variables'].items():
            print(f"  - {name}: {details['type']}, shape: {details['shape']}")
    else:
        print("  None")
    
    print(f"\n DETERMINISTIC VARIABLES:")
    if info['deterministic_variables']:
        for name, details in info['deterministic_variables'].items():
            print(f"  - {name}: {details['type']}, shape: {details['shape']}")
    else:
        print("  None")
    
    print(f"\n DATA VARIABLES (pm.Data containers):")
    if info['data_variables']:
        for name, details in info['data_variables'].items():
            print(f"  - {name}: {details['type']}, shape: {details['shape']}")
    else:
        print("  None (model doesn't use pm.Data containers)")
    
    print(f"\n COUNTERFACTUAL ANALYSIS:")
    if info['data_variables']:
        print("   Model has pm.Data containers - can use pm.set_data() approach")
    else:
        print("   Model doesn't have pm.Data containers - will use alternative approach")
        print("   Alternative approach works with generic beta coefficients")
    
    print(f"{'='*60}\n")


def get_model_structure(model: pm.Model) -> Dict[str, Any]:
    """
    Extract model structure information for diagnostics and visualization.
    
    This function provides compatibility with existing workflow functions
    by extracting key model information in a standardized format.
    
    Parameters
    ----------
    model : pm.Model
        The PyMC model to analyze
        
    Returns
    -------
    dict
        Dictionary with model structure information
    """
    structure = {
        "free_vars": [rv.name for rv in model.free_RVs],
        "observed_vars": [rv.name for rv in model.observed_RVs],
        "coords": getattr(model, 'coords', {}),
        "name": getattr(model, 'name', 'unnamed_model')
    }
    
    return structure
