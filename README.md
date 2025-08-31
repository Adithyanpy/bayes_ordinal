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

## Foundation & Acknowledgments

This package is built upon and extends the excellent PyMC examples and documentation, as well as foundational research in Bayesian workflow:

- **[Statistical Rethinking: A Bayesian Course with Examples in R and Stan (McElreath, 2020)](https://xcelab.net/rm/statistical-rethinking/)** - The definitive guide to Bayesian data analysis and workflow, providing the philosophical foundation, practical methodology, and diagnostic approaches that inform our package's design. Many of our workflow components, diagnostic tools, and model comparison methods are inspired by the principles and examples in this book.

- **[PyMC Ordered Categories Example](https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/11-Ordered_Categories.html)** - Statistical Rethinking lecture series on ordered categorical data, providing the foundational understanding of cumulative ordinal regression models and their implementation in PyMC.

- **[PyMC GLM Ordinal Regression Example](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-ordinal-regression.html)** - Comprehensive guide to generalized linear models for ordinal regression, offering insights into model specification, prior choices, and diagnostic approaches.

- **[PyMC Markov Chain Monte Carlo Example](https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/08-Markov_Chain_Monte_Carlo.html)** - Statistical Rethinking lecture on MCMC diagnostics and computational issue resolution, providing the foundation for our comprehensive diagnostic tools, convergence assessment, and computational problem-solving approaches.

- **[Bayesian Workflow (Gelman et al., 2020)](https://arxiv.org/pdf/2011.01808)** - Foundational paper that establishes the principles and practices of modern Bayesian workflow, including prior predictive checking, posterior predictive checking, cross-validation, model comparison, and computational problem resolution. Our package implements many of these workflow components as integrated, production-ready tools.

Our implementation builds upon these examples and research by providing a unified, production-ready interface that combines the best practices from all sources while adding comprehensive workflow tools, diagnostics, and analysis capabilities.


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



## Core Models

### Cumulative Model

The cumulative model implements proportional odds ordinal regression with flexible cutpoint approaches:

```python
model = bo.cumulative_model(
    y=y, X=X, K=4,
    link="logit",  # "logit", "probit"
    priors={"beta": [0, 1], "sigma": 1.0},
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
    kind="hist"  # "hist", "ecdf"
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
    hyper_name="sigma",
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
is_valid = bo.validate_ordinal_model(y, X, K)
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

- **pandas** - Data manipulation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## References

- McElreath, R. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and Stan.
- Bürkner, P. C., & Vuorre, M. (2019). Ordinal regression models in psychology: A tutorial.
- Gelman, A., et al. (2013). Bayesian Data Analysis.
- Fielding, A., Yang, M., & Goldstein, H. (2003). Multilevel ordinal models for examination grades. *Statistical Modelling*, 3(4), 339-355. [https://www.bristol.ac.uk/media-library/sites/cmm/migrated/documents/momeg.pdf](https://www.bristol.ac.uk/media-library/sites/cmm/migrated/documents/momeg.pdf)
- PyMC Documentation: https://docs.pymc.io/
- ArviZ Documentation: https://python.arviz.org/
