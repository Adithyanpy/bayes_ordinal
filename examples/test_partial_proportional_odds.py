#!/usr/bin/env python3
"""
Test Partial Proportional Odds for ASPIRES Data
==============================================

This script tests the proportional odds assumption and implements
partial proportional odds models to fix the LIFESAT issues.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load ASPIRES data and prepare for analysis"""
    print("🔍 Loading ASPIRES data...")
    
    data_path = Path("../Dataset/UKDA-9224-spss/spss/spss28/aspires3_survey.sav")
    df = pd.read_spss(data_path)
    
    print(f"✅ Data loaded: {df.shape}")
    
    # Focus on LIFESAT and key predictors
    lifesat_col = 'LIFESAT'
    
    # Clean LIFESAT - extract numeric values
    numeric_mapping = {}
    for val in df[lifesat_col].unique():
        if isinstance(val, str):
            if '-' in val:
                numeric_part = val.split('-')[0].strip()
                try:
                    numeric_val = int(numeric_part)
                    numeric_mapping[val] = numeric_val
                except ValueError:
                    pass
            else:
                try:
                    numeric_val = int(val)
                    numeric_mapping[val] = numeric_val
                except ValueError:
                    pass
        else:
            numeric_mapping[val] = val
    
    df['LIFESAT_CLEAN'] = df[lifesat_col].map(numeric_mapping)
    df_clean = df.dropna(subset=['LIFESAT_CLEAN'])
    
    print(f"✅ Cleaned data: {df_clean.shape}")
    
    # Select key predictors (based on your existing models)
    predictors = ['stem_identity', 'belonging', 'mental_health', 'female', 'ses', 'school_exp']
    
    # Check which predictors exist
    available_predictors = [col for col in predictors if col in df_clean.columns]
    print(f"✅ Available predictors: {available_predictors}")
    
    # Prepare X and y
    X = df_clean[available_predictors].values
    y = df_clean['LIFESAT_CLEAN'].values
    
    # Handle missing values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    print(f"✅ Final data: X={X.shape}, y={len(y)}")
    print(f"✅ LIFESAT range: {y.min()} to {y.max()}")
    print(f"✅ Unique categories: {len(np.unique(y))}")
    
    return X, y, available_predictors

def test_proportional_odds_assumption(X, y, feature_names):
    """Test proportional odds assumption for each predictor"""
    print(f"\n🔍 TESTING PROPORTIONAL ODDS ASSUMPTION")
    print("="*60)
    
    # Test each predictor individually
    results = {}
    
    for i, feature_name in enumerate(feature_names):
        print(f"\n--- Testing {feature_name} ---")
        
        # Get unique values for this predictor
        unique_vals = np.unique(X[:, i])
        if len(unique_vals) > 10:  # If continuous, create bins
            bins = np.percentile(X[:, i], np.linspace(0, 100, 6))
            X_binned = np.digitize(X[:, i], bins)
            unique_vals = np.unique(X_binned)
        else:
            X_binned = X[:, i]
        
        # Calculate cumulative proportions for each value
        cumulative_props = {}
        for val in unique_vals:
            mask = X_binned == val
            if mask.sum() > 10:  # Need sufficient samples
                y_subset = y[mask]
                for k in range(int(y.min()), int(y.max())):
                    cum_prop = (y_subset <= k).mean()
                    if val not in cumulative_props:
                        cumulative_props[val] = {}
                    cumulative_props[val][k] = cum_prop
        
        # Test if gaps are roughly equal across categories
        if len(cumulative_props) >= 2:
            # Calculate gaps for each category
            gaps_by_category = {}
            for k in range(int(y.min()), int(y.max())):
                gaps = []
                vals = sorted(cumulative_props.keys())
                for j in range(len(vals) - 1):
                    if k in cumulative_props[vals[j]] and k in cumulative_props[vals[j+1]]:
                        gap = cumulative_props[vals[j+1]][k] - cumulative_props[vals[j]][k]
                        gaps.append(gap)
                
                if gaps:
                    gaps_by_category[k] = gaps
                    gap_cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
                    print(f"  Category {k}: gap CV = {gap_cv:.3f}")
            
            # Overall assessment
            all_gaps = [gap for gaps in gaps_by_category.values() for gap in gaps]
            if all_gaps:
                overall_cv = np.std(all_gaps) / np.mean(all_gaps)
                print(f"  Overall gap CV: {overall_cv:.3f}")
                
                if overall_cv > 0.5:
                    status = "❌ VIOLATES proportional odds"
                elif overall_cv > 0.3:
                    status = "⚠️  QUESTIONABLE proportional odds"
                else:
                    status = "✅ REASONABLE proportional odds"
                
                results[feature_name] = {
                    'gap_cv': overall_cv,
                    'status': status,
                    'recommendation': 'nonproportional' if overall_cv > 0.5 else 'proportional'
                }
                print(f"  Status: {status}")
    
    return results

def build_partial_proportional_odds_model(X, y, feature_names, proportional_features, nonproportional_features):
    """Build partial proportional odds model using your existing implementation"""
    print(f"\n🔨 BUILDING PARTIAL PROPORTIONAL ODDS MODEL")
    print("="*60)
    
    # Import your partial odds model
    import sys
    sys.path.append('../bayes_ordinal')
    from models.partial_odds import partial_odds_model
    
    # Number of categories
    K = len(np.unique(y))
    print(f"Categories: {K}")
    print(f"Proportional features: {proportional_features}")
    print(f"Non-proportional features: {nonproportional_features}")
    
    # Build model
    model = partial_odds_model(
        y=y,
        X=X,
        K=K,
        proportional_features=proportional_features,
        nonproportional_features=nonproportional_features,
        link="logit",
        feature_names=feature_names,
        model_name="partial_proportional_odds"
    )
    
    return model

def fit_and_analyze_model(model, X, y):
    """Fit the model and analyze results"""
    print(f"\n🚀 FITTING MODEL")
    print("="*60)
    
    with model:
        # Sample
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.8,
            return_inferencedata=True
        )
    
    print(f"✅ Sampling completed")
    
    # Diagnostics
    print(f"\n📊 MODEL DIAGNOSTICS")
    print("="*60)
    
    # Summary
    summary = az.summary(trace)
    print(f"Model summary shape: {summary.shape}")
    
    # Check convergence
    max_rhat = summary['r_hat'].max()
    min_ess = summary['ess_bulk'].min()
    
    print(f"Max R-hat: {max_rhat:.3f}")
    print(f"Min ESS: {min_ess:.0f}")
    
    if max_rhat < 1.01 and min_ess > 1000:
        print("✅ Model converged well")
    else:
        print("⚠️  Convergence issues detected")
    
    return trace, summary

def compare_models(X, y, feature_names, test_results):
    """Compare different model specifications"""
    print(f"\n🔍 COMPARING MODEL SPECIFICATIONS")
    print("="*60)
    
    # Strategy 1: All proportional (standard ordinal)
    print(f"\n--- Strategy 1: All Proportional (Standard Ordinal) ---")
    proportional_features = list(range(len(feature_names)))
    nonproportional_features = []
    
    try:
        model1 = build_partial_proportional_odds_model(
            X, y, feature_names, proportional_features, nonproportional_features
        )
        trace1, summary1 = fit_and_analyze_model(model1, X, y)
        print("✅ Standard ordinal model completed")
    except Exception as e:
        print(f"❌ Standard ordinal failed: {e}")
        trace1, summary1 = None, None
    
    # Strategy 2: Mixed approach based on test results
    print(f"\n--- Strategy 2: Mixed Proportional/Non-Proportional ---")
    proportional_features = []
    nonproportional_features = []
    
    for i, feature_name in enumerate(feature_names):
        if feature_name in test_results:
            if test_results[feature_name]['recommendation'] == 'proportional':
                proportional_features.append(i)
            else:
                nonproportional_features.append(i)
        else:
            # Default to proportional if no test result
            proportional_features.append(i)
    
    print(f"Proportional features: {[feature_names[i] for i in proportional_features]}")
    print(f"Non-proportional features: {[feature_names[i] for i in nonproportional_features]}")
    
    try:
        model2 = build_partial_proportional_odds_model(
            X, y, feature_names, proportional_features, nonproportional_features
        )
        trace2, summary2 = fit_and_analyze_model(model2, X, y)
        print("✅ Mixed model completed")
    except Exception as e:
        print(f"❌ Mixed model failed: {e}")
        trace2, summary2 = None, None
    
    return {
        'standard_ordinal': (trace1, summary1),
        'mixed_model': (trace2, summary2)
    }

def main():
    """Main analysis function"""
    print("🔍 PARTIAL PROPORTIONAL ODDS ANALYSIS FOR ASPIRES DATA")
    print("="*60)
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data()
    
    # Test proportional odds assumption
    test_results = test_proportional_odds_assumption(X, y, feature_names)
    
    # Compare different model specifications
    model_results = compare_models(X, y, feature_names, test_results)
    
    # Summary and recommendations
    print(f"\n" + "="*60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    
    print(f"Feature analysis:")
    for feature_name, result in test_results.items():
        print(f"  {feature_name}: {result['status']}")
    
    print(f"\nModel comparison:")
    for model_name, (trace, summary) in model_results.items():
        if trace is not None and summary is not None:
            max_rhat = summary['r_hat'].max()
            min_ess = summary['ess_bulk'].min()
            print(f"  {model_name}: R-hat={max_rhat:.3f}, ESS={min_ess:.0f}")
        else:
            print(f"  {model_name}: Failed")
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
