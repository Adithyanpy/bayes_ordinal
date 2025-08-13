# Prior Predictive Checks - User Guide

This guide shows how to use the enhanced `run_prior_predictive` function for Bayesian ordinal models.

##  Quick Start

### Basic Usage

```python
from bayes_ordinal.models.cumulative import cumulative_model
from bayes_ordinal.workflow.prior_predictive import run_prior_predictive

# Build your model
model = cumulative_model(y, X, K, link="logit")

# Run prior predictive check (that's it!)
idata = run_prior_predictive(model, draws=1000, plot=True, y_obs=y)
```

### What Happens Automatically

 **Model Validation**: Checks model structure before sampling  
 **Variable Detection**: Automatically finds response variables  
 **Progress Tracking**: Shows step-by-step progress  
 **Error Handling**: Clear error messages if something goes wrong  
 **Comprehensive Plotting**: 6 informative plots showing prior distributions  
 **Result Validation**: Ensures results are reasonable  

##  Function Parameters

```python
run_prior_predictive(
    model,                    # Your PyMC model
    draws=200,               # Number of prior samples
    plot=True,               # Generate plots
    y_obs=None,              # Observed data for comparison
    model_name=None,         # Explicit model name (auto-detected if None)
    validate_model=True,     # Validate model before sampling
    verbose=True,            # Show progress messages
    plot_kwargs=None         # Custom plotting options
)
```

##  Use Cases

### 1. Simple Model Check

```python
# Just check if your priors make sense
model = cumulative_model(y, X, K, link="logit")
idata = run_prior_predictive(model)
```

### 2. Compare Different Models

```python
# Logit model
model_logit = cumulative_model(y, X, K, link="logit", model_name="logit_model")
idata_logit = run_prior_predictive(model_logit, draws=500, plot=True, y_obs=y)

# Probit model  
model_probit = cumulative_model(y, X, K, link="probit", model_name="probit_model")
idata_probit = run_prior_predictive(model_probit, draws=500, plot=True, y_obs=y)
```

### 3. Custom Prior Evaluation

```python
# Test different prior specifications
custom_priors = {"beta": [0, 3], "sigma": 2.0}
model = cumulative_model(y, X, K, link="logit", priors=custom_priors, model_name="wide_priors")
idata = run_prior_predictive(model, draws=1000, plot=True, y_obs=y)
```

### 4. Silent Mode (for scripts)

```python
# No progress messages, just results
idata = run_prior_predictive(
    model, 
    plot=False,      # No plots
    verbose=False    # No messages
)
```

##  What You Get

### Return Value
- **`idata`**: ArviZ InferenceData object with prior predictive samples
- **Access variables**: `idata.prior_predictive['model_name::y']`

### Automatic Plots (6 subplots)
1. **Sample Distributions**: Individual prior predictive samples
2. **Mean Distribution**: Average across all samples  
3. **Observed Comparison**: Your data vs. prior predictions
4. **Category Counts**: Distribution of counts per category
5. **Total Observations**: Distribution of sample sizes
6. **Category Proportions**: Proportions across samples

##  Advanced Features

### Custom Plotting

```python
# Customize plot appearance
idata = run_prior_predictive(
    model,
    plot=True,
    plot_kwargs={
        "figsize": (16, 12),
        "dpi": 100
    }
)
```

### Model Name Auto-Detection

```python
# The function automatically detects model names
model = cumulative_model(y, X, K, link="logit")  # No model_name needed
idata = run_prior_predictive(model)  # Works automatically!
```

### Error Recovery

```python
# If plotting fails, you still get the data
try:
    idata = run_prior_predictive(model, plot=True)
except RuntimeError as e:
    print(f"Plotting failed but data is available: {e}")
    # idata still contains your prior predictive samples
```

##  Complete Workflow Example

```python
from bayes_ordinal.models.cumulative import cumulative_model
from bayes_ordinal.workflow.prior_predictive import run_prior_predictive

# 1. Prepare data
X, y, K = your_data_preparation()

# 2. Build model
model = cumulative_model(y, X, K, link="logit", model_name="my_model")

# 3. Check priors
idata_prior = run_prior_predictive(model, draws=1000, plot=True, y_obs=y)

# 4. If priors look good, run posterior sampling
with model:
    idata_posterior = pm.sample(2000, tune=1000)

# 5. Continue with your analysis...
```

##  Troubleshooting

### Common Issues

**"Could not identify response variable"**
- Solution: Use explicit `model_name` parameter
- Or ensure your model has a clear response variable

**"Model has no observed variables"**
- Solution: Check that your model includes the likelihood (e.g., `pm.OrderedLogistic`)

**"Plotting failed"**
- Solution: Check matplotlib backend, or set `plot=False` to get data only

### Getting Help

1. **Check verbose output**: Set `verbose=True` for detailed progress
2. **Validate model**: Set `validate_model=True` to catch issues early
3. **Use explicit names**: Specify `model_name` for complex models

##  Best Practices

1. **Start simple**: Use default parameters first
2. **Check priors**: Always run prior predictive before posterior sampling
3. **Use meaningful names**: Give models descriptive names for easier debugging
4. **Compare models**: Test different prior specifications
5. **Validate results**: Check that prior predictions look reasonable

##  Related Functions

- **`cumulative_model()`**: Build cumulative ordinal models
- **`pm.sample()`**: Run posterior sampling
- **`az.plot_trace()`**: Check MCMC diagnostics
- **`az.summary()`**: Get parameter summaries

---

**Need more help?** Check the examples in the `examples/` directory or run the test scripts to see the function in action!
