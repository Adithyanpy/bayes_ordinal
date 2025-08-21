# Bayesian Ordinal Regression Package

A comprehensive PyMC-based package for Bayesian ordinal regression with advanced workflow tools, diagnostics, and analysis capabilities.

## Overview

This package provides a complete Bayesian workflow for ordinal regression analysis, featuring:

- **Advanced Models**: Cumulative (proportional odds) models with flexible cutpoint approaches
- **Comprehensive Workflow**: Prior predictive checks, model fitting, diagnostics, and posterior predictive analysis
- **Robust Diagnostics**: MCMC convergence, computational issue detection, and model validation
- **Model Comparison**: Cross-validation with LOO/WAIC, stacking, and interpretation tools
- **Hierarchical Support**: Group-level varying intercepts and random effects
- **Counterfactual Analysis**: Scenario-based predictions and causal inference
- **Configuration System**: Flexible configuration management for different analysis needs

## Installation

```bash
pip install bayes-ordinal
```

Or install from source:

```bash
git clone https://github.com/Adithyanpy/bayes_ordinal.git
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

# Build model
model = bo.cumulative_model(y=y, X=X, K=K, link="logit")

# Run prior predictive check
prior_idata = bo.run_prior_predictive(model, draws=200, plot=True)

# Fit model with optimized parameters
idata = bo.fit_ordinal_model(model, draws=2000, tune=1000, chains=4)

# Comprehensive diagnostics
diagnostics = bo.run_comprehensive_diagnostics(idata)
summary = bo.create_model_summary(idata)

# Posterior predictive checks
ppc = bo.run_posterior_predictive(model, idata)
```

### Advanced Workflow with Configuration

```python
from bayes_ordinal.config import get_production_config, Config

# Use preset configurations
config = get_production_config()
config.model.model_type = "cumulative"
config.model.link = "logit"
config.sampling.draws = 2000

# Or create custom configuration
custom_config = Config()
custom_config.name = "my_analysis"
custom_config.sampling.draws = 2000
custom_config.workflow.run_sensitivity_analysis = True

# Build and run analysis
model = bo.cumulative_model(y=y, X=X, K=K)
prior_idata = bo.run_prior_predictive(model, draws=200)
idata = bo.fit_ordinal_model(model, draws=custom_config.sampling.draws)
ppc = bo.run_posterior_predictive(model, idata)
```

## Core Models

### Cumulative Model

The cumulative model implements proportional odds ordinal regression with flexible cutpoint approaches:

```python
model = bo.cumulative_model(
    y=y, X=X, K=4,
    link="logit",  # "logit", "probit", "cloglog"
    priors={"coef_mu": 0.0, "coef_sigma": 2.5},
    prior_type="exponential_sigma",  # "fixed_sigma" or "exponential_sigma"
    feature_names=["Age", "Income", "Education"]
)
```

**Key Features:**
- **Flexible Cutpoints**: Supports both constrained (Dirichlet) and flexible (Normal) approaches
- **Hierarchical Structure**: Group-level varying intercepts with `group_idx` and `n_groups`
- **Automatic Scaling**: Probit models automatically adjust prior scales
- **Validation**: Comprehensive input validation and error handling

## Workflow Components

### 1. Prior Predictive Analysis

```python
# Basic prior predictive check
idata = bo.run_prior_predictive(model, draws=1000, plot=True, y_obs=y)

# Custom plots
custom_plots = {
    'prior_samples': True,      # Prior predictive samples
    'mean_distribution': True,  # Mean distribution
    'observed': True,           # Observed data comparison
    'category_counts': True,    # Category counts per sample
    'total_observations': True, # Total observations per sample
    'category_proportions': True # Category proportions across samples
}

idata = bo.run_prior_predictive(
    model, draws=1000, y_obs=y, custom_plots=custom_plots
)
```

### 2. Model Fitting

```python
# Optimized fitting for ordinal models
idata = bo.fit_ordinal_model(
    model,
    draws=2000,        # Recommended for ordinal models
    tune=1000,         # Proper NUTS adaptation
    chains=4,          # Reliable inference
    target_accept=0.8, # Optimal for constrained parameters
    max_treedepth=15   # Higher for complex models
)
```

### 3. Comprehensive Diagnostics

```python
# Full diagnostic suite
diagnostics = bo.run_comprehensive_diagnostics(idata)

# Individual diagnostic components
summary = bo.summarize_diagnostics(idata)
plots = bo.plot_diagnostics(idata)
forest = bo.plot_group_forest(idata, var_name="u")
model_summary = bo.create_model_summary(idata)
```

### 4. Posterior Predictive Analysis

```python
# Run posterior predictive checks
ppc = bo.run_posterior_predictive(
    model, idata,
    kind="proportions"  # "proportions", "hist", "ecdf"
)
```

### 5. Model Comparison

```python
# Compare multiple models
models = {"model1": model1, "model2": model2}
idatas = {"model1": idata1, "model2": idata2}

# Basic comparison
comparison = bo.compare_models(models, idatas, ic="loo")

# Advanced comparison with stacking
stacking_results = bo.compare_models_stacking(models, idatas)

# Interpretable comparison
interpretation = bo.compare_models_interpretation(models, idatas)
bo.plot_model_comparison_interpretation(interpretation)
```

### 6. Computational Issue Resolution

```python
# Diagnose and resolve computational issues
issues = bo.diagnose_computational_issues(idata)

# Check for multimodality
multimodality = bo.check_multimodality(idata)

# Stack individual chains if needed
stacked_idata = bo.stack_individual_chains(idata)

# Comprehensive computation check
computation_check = bo.comprehensive_computation_check(idata)
```

### 7. Sensitivity Analysis

```python
# Prior sensitivity analysis
sensitivity = bo.prior_sensitivity(
    bo.cumulative_model,
    y=y, X=X, K=4,
    hyper_name="coef_sigma",
    hyper_values=[1.0, 2.5, 5.0],
    trace_var="beta"
)

# Influence diagnostics
influential = bo.plot_influential(idata, threshold=0.7)
```

## Data Processing & Utilities

### Data Validation

```python
# Validate ordinal data
y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K=4)

# Encode categorical features
X_encoded, encoders = bo.encode_categorical_features(X, categorical_cols=["region"])

# Standardize features
X_scaled, scaler = bo.standardize_features(X)

# Create group indices for hierarchical models
group_idx, n_groups, group_map = bo.create_group_indices(group_variable)

# Compute category proportions
proportions = bo.compute_category_proportions(y, K)
```

### Model Inspection

```python
# Inspect model variables
variables = bo.inspect_model_variables(model)

# Print model summary
bo.print_model_summary(model)

# Validate ordinal model structure
is_valid = bo.validate_ordinal_model(model)
```

## Counterfactual Analysis

```python
# Define scenarios
scenarios = {
    "baseline": {"action": 0, "intention": 0, "contact": 0},
    "high_risk": {"action": 1, "intention": 1, "contact": 1}
}

# Run counterfactual analysis
results = bo.run_counterfactual_analysis(
    model, idata, scenarios, feature_names, n_samples=1000
)

# Plot results
bo.plot_counterfactual_results(results)
```

## Configuration System

The package provides a flexible configuration system:

```python
from bayes_ordinal.config import (
    Config, ModelConfig, PriorConfig, SamplingConfig, 
    WorkflowConfig, DataConfig, OutputConfig,
    get_default_config, get_quick_test_config, 
    get_production_config, get_hierarchical_config
)

# Preset configurations
config = get_production_config()        # Production settings
config = get_quick_test_config()        # Quick testing
config = get_hierarchical_config()      # Hierarchical models

# Custom configuration
config = Config()
config.model.model_type = "cumulative"
config.model.link = "logit"
config.sampling.draws = 2000
config.workflow.run_sensitivity_analysis = True
```

## Complete Analysis Example

```python
import numpy as np
import bayes_ordinal as bo

# 1. Prepare and validate data
np.random.seed(42)
n, K = 200, 5
X = np.random.normal(size=(n, 3))
y = np.random.randint(0, K, size=n)

y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K)

# 2. Build models
models = {
    "logit": bo.cumulative_model(y=y_clean, X=X_clean, K=K, link="logit"),
    "probit": bo.cumulative_model(y=y_clean, X=X_clean, K=K, link="probit")
}

# 3. Complete workflow for each model
idatas = {}
for name, model in models.items():
    print(f"\n=== {name.upper()} MODEL ===")
    
    # Prior predictive check
    prior_idata = bo.run_prior_predictive(model, draws=200, y_obs=y_clean)
    
    # Fit model
    idata = bo.fit_ordinal_model(model, draws=2000, tune=1000, chains=4)
    idatas[name] = idata
    
    # Comprehensive diagnostics
    diagnostics = bo.run_comprehensive_diagnostics(idata)
    summary = bo.create_model_summary(idata)
    
    # Posterior predictive check
    ppc = bo.run_posterior_predictive(model, idata)

# 4. Model comparison
comparison = bo.compare_models(models, idatas, ic="loo")
stacking_results = bo.compare_models_stacking(models, idatas)
interpretation = bo.compare_models_interpretation(models, idatas)

# 5. Display results
bo.display_comparison_results(comparison)
bo.plot_model_comparison_interpretation(interpretation)

# 6. Check convergence
for name, idata in idatas.items():
    convergence = bo.check_convergence(idata)
    print(f"{name}: Converged = {convergence['converged']}")
    
    diag_df = bo.summarize_diagnostics(idata)
    print(f"{name} R-hat max: {diag_df['r_hat'].max():.4f}")
    print(f"{name} ESS min: {diag_df['ess_bulk'].min():.0f}")
```

## Dependencies

- **PyMC** >= 5.3 - Probabilistic programming
- **ArviZ** >= 0.16 - Bayesian inference diagnostics
- **NumPy** >= 1.24 - Numerical computing
- **SciPy** >= 1.10 - Scientific computing
- **Matplotlib** >= 3.6 - Plotting
- **scikit-learn** - Data preprocessing
- **PyYAML** >= 6.0 - Configuration files
- **pandas** - Data manipulation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- McElreath, R. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and Stan.
- Bürkner, P. C., & Vuorre, M. (2019). Ordinal regression models in psychology: A tutorial.
- Gelman, A., et al. (2013). Bayesian Data Analysis.
- Fielding, A., Yang, M., & Goldstein, H. (2003). Multilevel ordinal models for examination grades. *Statistical Modelling*, 3(4), 339-355. [https://www.bristol.ac.uk/media-library/sites/cmm/migrated/documents/momeg.pdf](https://www.bristol.ac.uk/media-library/sites/cmm/migrated/documents/momeg.pdf)
- PyMC Documentation: https://docs.pymc.io/
- ArviZ Documentation: https://python.arviz.org/
