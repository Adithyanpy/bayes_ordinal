"""
PyMC Documentation-Compatible Example

This example demonstrates how to use the updated bayes_ordinal package
to follow the PyMC Ordered Categories documentation exactly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pymc as pm
import bayes_ordinal as bo

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

print("✓ PyMC Documentation-Compatible Example")
print("✓ Using updated bayes_ordinal package")

# Load the trolley problem dataset
print("\n1. Loading Data...")
data = pd.read_csv('../Dataset/Trolley.csv', sep=';')

# Clean the data
data['response'] = data['response'].str.strip('"').astype(int)
data = data.dropna()
data = data[data['response'].between(1, 7)]

print(f"✓ Dataset shape: {data.shape}")
print(f"✓ Response range: {data['response'].min()} to {data['response'].max()}")

# Define feature columns
feature_cols = ['action', 'intention', 'contact']
feature_names = ['Action', 'Intention', 'Contact']

print(f"✓ Features: {feature_names}")

# Build model following PyMC documentation exactly
print("\n2. Building Model (PyMC Documentation Style)...")
model = bo.cumulative_model(
    y=data['response'],
    X=data[feature_cols],
    link="logit",
    model_name="trolley_logit"
)

print("✓ Model built successfully")
print(f"✓ Model variables: {list(model.named_vars.keys())}")

# Display model structure
print("\n3. Model Structure...")
bo.plot_model_structure(model)

# Fit the model
print("\n4. Fitting Model...")
with model:
    # Set the data for fitting
    pm.set_data({
        "action": data['action'].astype(float),
        "intention": data['intention'].astype(float),
        "contact": data['contact'].astype(float)
    })
    
    idata = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        return_inferencedata=True,
        random_seed=42
    )

print("✓ Model fitting completed")
print(f"✓ Posterior samples shape: {idata.posterior.dims}")

# Diagnostics
print("\n5. Model Diagnostics...")
conv_check = bo.check_convergence(idata)
print(f"✓ Converged: {conv_check['converged']}")

diag = bo.summarize_diagnostics(idata)
print("\nDiagnostics Summary:")
print(diag[['r_hat', 'ess_bulk', 'ess_tail', 'n_divergences']].round(3))

# Plot diagnostics
az.plot_trace(idata, var_names=["beta_action", "beta_intention", "beta_contact", "alpha"])
plt.tight_layout()
plt.show()

# Parameter visualization
print("\n6. Parameter Visualization...")
az.plot_forest(idata, var_names=["beta_action", "beta_intention", "beta_contact"], 
               combined=True, figsize=(8, 4))
plt.title('Coefficient Effects')
plt.show()

az.plot_forest(idata, var_names=["alpha"], 
               combined=True, figsize=(8, 4))
plt.title('Cutpoints')
plt.show()

# Counterfactual analysis
print("\n7. Counterfactual Analysis...")
scenarios = {
    'Baseline': np.array([0, 0, 0]),
    'High Action': np.array([1, 0, 0]),
    'High Intention': np.array([0, 1, 0]),
    'High Contact': np.array([0, 0, 1]),
    'Action + Intention': np.array([1, 1, 0]),
    'All High': np.array([1, 1, 1])
}

counterfactual_results = bo.run_counterfactual_analysis(
    model=model,
    idata=idata,
    scenarios=scenarios,
    feature_names=feature_cols,
    n_samples=1000
)

print("✓ Counterfactual analysis completed")

# Plot counterfactual results
bo.plot_counterfactual_results(counterfactual_results)

# Print counterfactual summary
print("\nCounterfactual Analysis Summary:")
for scenario_name, result in counterfactual_results.items():
    print(f"  {scenario_name}:")
    print(f"    Mean: {result['mean']:.2f}")
    print(f"    Std: {result['std']:.2f}")
    print(f"    95% CI: [{result['percentiles'][0]:.1f}, {result['percentiles'][4]:.1f}]")

# Posterior predictive checks
print("\n8. Posterior Predictive Checks...")
with model:
    pm.set_data({
        "action": data['action'].astype(float),
        "intention": data['intention'].astype(float),
        "contact": data['contact'].astype(float)
    })
    ppc = pm.sample_posterior_predictive(idata, return_inferencedata=True)

print("✓ Posterior predictive checks completed")

# Compare observed vs predicted
observed = data['response'].values
predicted = ppc.posterior_predictive['response'].values.flatten() + 1

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(observed, bins=range(1, 9), alpha=0.7, label='Observed', 
         density=True, color='lightcoral', edgecolor='black')
plt.hist(predicted, bins=range(1, 9), alpha=0.7, label='Predicted', 
         density=True, color='skyblue', edgecolor='black')
plt.xlabel('Response Category')
plt.ylabel('Density')
plt.title('Observed vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
observed_counts = pd.Series(observed).value_counts().sort_index()
predicted_counts = pd.Series(predicted).value_counts().sort_index()
plt.bar(observed_counts.index - 0.2, observed_counts.values, alpha=0.7, 
        label='Observed', color='lightcoral', width=0.4)
plt.bar(predicted_counts.index + 0.2, predicted_counts.values, alpha=0.7, 
        label='Predicted', color='skyblue', width=0.4)
plt.xlabel('Response Category')
plt.ylabel('Count')
plt.title('Observed vs Predicted Counts')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Results summary
print("\n9. Results Summary...")
print("=" * 50)
print(f"Dataset: {data.shape[0]} observations, {len(feature_names)} features")
print(f"Response categories: {data['response'].min()}-{data['response'].max()}")
print(f"Model: OrderedLogistic with {len(feature_names)} coefficients")
print(f"Convergence: {conv_check['converged']}")
print(f"R-hat range: {diag['r_hat'].min():.3f} to {diag['r_hat'].max():.3f}")

print("\n✓ PyMC Documentation-Compatible Example Completed!")
print("✓ Package now follows PyMC best practices exactly") 