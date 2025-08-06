# API Reference

## Models

### Cumulative Model

```python
bayes_ordinal.cumulative_model(y, X, K, priors=None, *, link="logit", constrained_uniform=False, group_idx=None, n_groups=None, model_name="cumulative_model")
```

Constructs a proportional-odds ordinal regression model using Dirichlet cutpoints.

**Parameters:**
- `y`: Ordinal response variable (0 to K-1)
- `X`: Covariate matrix
- `K`: Number of categories
- `priors`: Dictionary of prior hyperparameters
- `link`: Link function ("logit", "probit", "cloglog")
- `constrained_uniform`: Use constrained Dirichlet cutpoints
- `group_idx`: Group indices for hierarchical model
- `n_groups`: Number of groups

### Partial Odds Model

```python
bayes_ordinal.partial_odds_model(y, X, K, priors=None, group_idx=None, n_groups=None)
```

Constructs a partial-odds ordinal regression model using adjacent-category formulation.

## Workflow Functions

### Model Fitting

```python
bayes_ordinal.fit_model(model, draws=1000, tune=1000, chains=4, return_inferencedata=True, **kwargs)
```

Fit a PyMC model using advanced sampling methods.

### Diagnostics

```python
bayes_ordinal.summarize_diagnostics(idata)
bayes_ordinal.check_convergence(idata)
```

### Predictive Checks

```python
bayes_ordinal.run_prior_predictive(model, draws=200, plot=True)
bayes_ordinal.run_posterior_predictive(model, idata, kind="proportions")
```

### Model Comparison

```python
bayes_ordinal.compare_models(models, idatas, ic="loo")
```

## Utilities

### Data Validation

```python
bayes_ordinal.validate_ordinal_data(y, X, K=None)
bayes_ordinal.encode_categorical_features(X, categorical_cols)
bayes_ordinal.standardize_features(X, scaler=None)
```

### Priors

```python
bayes_ordinal.get_default_priors()
bayes_ordinal.get_weakly_informative_priors()
bayes_ordinal.get_informative_priors(coef_means, coef_sds)
```

### Visualization

```python
bayes_ordinal.plot_model_comparison(idatas, ic="loo")
bayes_ordinal.plot_cutpoints(idata, var_name="alpha")
bayes_ordinal.plot_coefficient_effects(idata, var_name="beta", feature_names=None)
```

## Configuration

```python
bayes_ordinal.Config()
bayes_ordinal.get_default_config()
bayes_ordinal.get_production_config()
bayes_ordinal.get_quick_test_config()
```
