# Bayesian Ordinal Regression Documentation

Welcome to the Bayesian Ordinal Regression package documentation. This package provides comprehensive tools for Bayesian ordinal regression analysis using PyMC.

## Quick Start

```python
import numpy as np
import bayes_ordinal as bo

# Generate example data
np.random.seed(42)
n, K = 100, 4
X = np.random.normal(size=(n, 2))
y = np.random.randint(0, K, size=n)

# Run complete workflow
results = bo.run_workflow(
    model_fn={"cumulative": bo.cumulative_model, "partial": bo.partial_odds_model},
    y=y, X=X, K=K,
    config={"draws": 1000, "tune": 1000, "chains": 4}
)
```

## Key Features

- **Multiple Model Types**: Cumulative (proportional odds) and partial odds models
- **Complete Workflow**: From prior specification to model comparison
- **Robust Diagnostics**: MCMC convergence, posterior predictive checks
- **Visualization Tools**: Specialized plots for ordinal regression
- **Hierarchical Support**: Group-level varying intercepts

## Documentation Sections

- [API Reference](api.md) - Complete API documentation
- [Workflow Guide](workflow.md) - Step-by-step workflow instructions
- [Examples](../examples/) - Jupyter notebooks and example scripts

## Installation

```bash
pip install bayes-ordinal
```

## Dependencies

- PyMC >= 5.3
- ArviZ >= 0.16
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.6
- scikit-learn (for preprocessing)
- PyYAML >= 6.0 (for configuration files)
