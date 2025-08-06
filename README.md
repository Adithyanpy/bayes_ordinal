# Bayesian Ordinal Regression

A comprehensive PyMC-based package for Bayesian ordinal regression with a complete workflow for analysis, visualization, and model verification.

## Overview

This package implements Bayesian ordinal regression models with a focus on usability and best practices. It provides:

- **Multiple model types**: Cumulative (proportional odds) and partial odds models
- **Complete workflow**: From prior specification to model comparison
- **Robust diagnostics**: MCMC convergence, posterior predictive checks, and sensitivity analysis
- **Visualization tools**: Specialized plots for ordinal regression analysis
- **Hierarchical support**: Group-level varying intercepts

## Installation

```bash
pip install bayes-ordinal
```

Or install from source:

```bash
git clone <repository-url>
cd bayes_ordinal
pip install -e .
```

## Quick Start

### Basic Usage

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

# View results
print(results["diagnostics"]["cumulative"])
```

### Using Configuration System

```python
from bayes_ordinal.config import get_production_config, Config

# Use preset configuration
config = get_production_config()
config.model.model_type = "cumulative"
config.model.link = "logit"

# Or create custom configuration
custom_config = Config()
custom_config.name = "my_analysis"
custom_config.sampling.draws = 2000
custom_config.workflow.run_sensitivity_analysis = True

# Run complete workflow
results = bo.run_workflow(
    model_fn=bo.cumulative_model,
    y=y, X=X, K=K,
    config=custom_config.get_workflow_config()
)
```

### Command Line Interface

```bash
# Run with default configuration
bayes-ordinal data.csv -o results/

# Run with custom configuration
bayes-ordinal data.csv -c config.yaml -o results/

# Quick test run
bayes-ordinal data.csv --quick-test -o results/

# Production run
bayes-ordinal data.csv --production -o results/

# Create configuration template
bayes-ordinal --create-config config_template.yaml
```

## Models

### Cumulative Model (Proportional Odds)

The cumulative model assumes proportional odds across categories:

```python
model = bo.cumulative_model(
    y=y, X=X, K=4,
    link="logit",  # or "probit", "cloglog"
    priors=bo.get_default_priors()
)
```

### Partial Odds Model

The partial odds model allows effects to vary by category:

```python
model = bo.partial_odds_model(
    y=y, X=X, K=4,
    priors=bo.get_default_priors()
)
```

## Workflow Components

### 1. Prior Specification

```python
# Default priors
priors = bo.get_default_priors()

# Weakly informative priors
priors = bo.get_weakly_informative_priors()

# Custom informative priors
priors = bo.get_informative_priors(
    coef_means=np.array([0.5, -0.3]),
    coef_sds=np.array([1.0, 1.0])
)
```

### 2. Prior Predictive Checks

```python
idata_prior = bo.run_prior_predictive(model, draws=200, plot=True)
```

### 3. Model Fitting

```python
idata = bo.fit_model(
    model,
    draws=2000,
    tune=1000,
    chains=4,
    smoke_test=False
)
```

### 4. Diagnostics

```python
# MCMC diagnostics
diag_df = bo.summarize_diagnostics(idata)

# Convergence check
conv_check = bo.check_convergence(idata)
print(f"Converged: {conv_check['converged']}")
```

### 5. Posterior Predictive Checks

```python
ppc = bo.run_posterior_predictive(
    model, idata,
    kind="proportions"  # or "hist", "ecdf"
)
```

### 6. Model Comparison

```python
comparison = bo.compare_models(
    {"cum": model1, "partial": model2},
    {"cum": idata1, "partial": idata2},
    ic="loo"
)
```

## Visualization

### Coefficient Effects

```python
bo.plot_coefficient_effects(
    idata,
    var_name="beta",
    feature_names=["Age", "Income"]
)
```

### Cutpoint Estimates

```python
bo.plot_cutpoints(idata, var_name="alpha")
```

### Model Comparison

```python
bo.plot_model_comparison(
    {"cumulative": idata1, "partial": idata2},
    ic="loo"
)
```

## Hierarchical Models

For grouped data with varying intercepts:

```python
# Create group indices
group_idx, n_groups, group_map = bo.create_group_indices(group_variable)

# Fit hierarchical model
model = bo.cumulative_model(
    y=y, X=X, K=K,
    group_idx=group_idx,
    n_groups=n_groups
)

# Plot group effects
bo.plot_group_forest(idata, var_name="u")
```

## Data Preprocessing

```python
# Validate and clean data
y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K=4)

# Encode categorical features
X_encoded, encoders = bo.encode_categorical_features(X, categorical_cols=["region"])

# Standardize features
X_scaled, scaler = bo.standardize_features(X)
```

## Sensitivity Analysis

```python
# Prior sensitivity
bo.prior_sensitivity(
    bo.cumulative_model,
    y=y, X=X, K=4,
    hyper_name="coef_sigma",
    hyper_values=[1.0, 2.5, 5.0],
    trace_var="beta"
)

# Influence diagnostics
bo.plot_influential(idata, threshold=0.7)
```

## Complete Example

```python
import numpy as np
import bayes_ordinal as bo

# 1. Prepare data
np.random.seed(42)
n, K = 200, 5
X = np.random.normal(size=(n, 3))
y = np.random.randint(0, K, size=n)

# 2. Validate data
y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K)

# 3. Run complete workflow
results = bo.run_workflow(
    model_fn={
        "cumulative": bo.cumulative_model,
        "partial": bo.partial_odds_model
    },
    y=y_clean, X=X_clean, K=K,
    config={
        "draws": 2000,
        "tune": 1000,
        "chains": 4,
        "run_prior": True,
        "run_ppc": True,
        "run_diagnostics": True,
        "run_cv": True
    }
)

# 4. Examine results
print("Model Comparison:")
print(results["cv"])

print("\nDiagnostics:")
print(results["diagnostics"]["cumulative"][["r_hat", "ess_bulk", "n_divergences"]])
```

## Dependencies

- PyMC >= 5.3
- ArviZ >= 0.16
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.6
- scikit-learn (for preprocessing)
- PyYAML >= 6.0 (for configuration files)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- McElreath, R. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and Stan.
- Bürkner, P. C., & Vuorre, M. (2019). Ordinal regression models in psychology: A tutorial.
- Gelman, A., et al. (2013). Bayesian Data Analysis.
