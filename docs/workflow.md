# Workflow Guide

## Complete Bayesian Workflow

The Bayesian ordinal regression package provides a comprehensive workflow for analysis:

### 1. Data Preparation

```python
import numpy as np
import bayes_ordinal as bo

# Load and validate data
y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K=4)

# Encode categorical features
X_encoded, encoders = bo.encode_categorical_features(X, ['region'])

# Standardize features
X_scaled, scaler = bo.standardize_features(X)
```

### 2. Prior Specification

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

### 3. Model Building

```python
# Cumulative model
cumulative_model = bo.cumulative_model(
    y=y, X=X, K=K,
    link="logit",
    priors=priors
)

# Partial odds model
partial_model = bo.partial_odds_model(
    y=y, X=X, K=K,
    priors=priors
)
```

### 4. Prior Predictive Checks

```python
# Run prior predictive sampling
idata_prior = bo.run_prior_predictive(
    cumulative_model,
    draws=200,
    plot=True
)
```

### 5. Model Fitting

```python
# Fit model
with cumulative_model:
    idata = bo.fit_model(
        cumulative_model,
        draws=2000,
        tune=1000,
        chains=4
    )
```

### 6. Diagnostics

```python
# Check convergence
conv_check = bo.check_convergence(idata)
print(f"Converged: {conv_check['converged']}")

# Summarize diagnostics
diag_df = bo.summarize_diagnostics(idata)
print(diag_df[['r_hat', 'ess_bulk', 'n_divergences']])
```

### 7. Posterior Predictive Checks

```python
# Run posterior predictive checks
ppc = bo.run_posterior_predictive(
    cumulative_model,
    idata,
    kind="proportions"
)
```

### 8. Model Comparison

```python
# Compare multiple models
models = {
    "cumulative": cumulative_model,
    "partial": partial_model
}
idatas = {
    "cumulative": idata_cum,
    "partial": idata_partial
}

comparison = bo.compare_models(models, idatas, ic="loo")
```

### 9. Visualization

```python
# Plot results
bo.plot_model_comparison(idatas, ic="loo")
bo.plot_cutpoints(idata, var_name="alpha")
bo.plot_coefficient_effects(idata, var_name="beta", feature_names=["Age", "Income"])
```

## Using the Complete Workflow

```python
# Run complete workflow
results = bo.run_workflow(
    model_fn={
        "cumulative": bo.cumulative_model,
        "partial": bo.partial_odds_model
    },
    y=y, X=X, K=K,
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
```

## Hierarchical Models

For grouped data:

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
