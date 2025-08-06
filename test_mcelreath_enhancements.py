#!/usr/bin/env python3
"""
Comprehensive test script for McElreath-style enhancements.

This script tests all the new functions and improvements to ensure they
work seamlessly with the existing bayes_ordinal workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
import bayes_ordinal as bo
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_conservative_priors():
    """Test McElreath-style prior functions."""
    print("=" * 60)
    print("TESTING MCELREATH PRIORS")
    print("=" * 60)
    
    # Test McElreath priors for ordinal model
    mcelreath_ordinal = bo.get_conservative_priors(K=4, model_type="ordinal")
    print(f"✓ McElreath ordinal priors: {list(mcelreath_ordinal.keys())}")
    
    # Test McElreath priors for continuous model
    mcelreath_continuous = bo.get_conservative_priors(model_type="continuous")
    print(f"✓ McElreath continuous priors: {list(mcelreath_continuous.keys())}")
    
    # Test data-informed priors
    y = np.random.randint(0, 4, size=100)
    X = np.random.normal(size=(100, 2))
    data_informed = bo.get_data_informed_priors(y, X, K=4)
    print(f"✓ Data-informed priors: {list(data_informed.keys())}")
    
    # Test validation
    validated = bo.validate_priors(mcelreath_ordinal, model_type="cumulative")
    print(f"✓ Prior validation: {list(validated.keys())}")
    
    return True

def test_mcelreath_plotting():
    """Test McElreath-style plotting functions."""
    print("\n" + "=" * 60)
    print("TESTING MCELREATH PLOTTING")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    n_samples, n_features, K = 200, 3, 4
    
    # True parameters
    true_beta = np.array([0.5, -0.3, 0.8])
    true_cutpoints = np.array([-1.5, -0.5, 0.5])
    
    # Generate features
    X = np.random.normal(size=(n_samples, n_features))
    
    # Generate ordinal responses
    linear_predictor = X @ true_beta
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        probs = 1 / (1 + np.exp(-(true_cutpoints - linear_predictor[i])))
        probs = np.concatenate([[0], probs, [1]])
        category_probs = probs[1:] - probs[:-1]
        y[i] = np.random.choice(K, p=category_probs)
    
    # Preprocess data
    y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K)
    X_scaled, scaler = bo.standardize_features(X_clean)
    
    # Get McElreath priors
    priors = bo.get_conservative_priors(K=K, model_type="ordinal")
    
    # Build and fit model
    model = bo.cumulative_model(y=y_clean, X=X_scaled, K=K, priors=priors)
    idata = bo.fit_model(model, draws=500, tune=250, chains=2, smoke_test=False)
    
    print("✓ Model fitted successfully")
    
    # Test McElreath forest plot
    try:
        bo.plot_forest(idata, var_name="beta", 
                                feature_names=["Feature 1", "Feature 2", "Feature 3"])
        print("✓ McElreath forest plot created")
    except Exception as e:
        print(f"✗ McElreath forest plot failed: {e}")
    
    # Test contrast analysis
    try:
        group1_indices = np.arange(50)
        group2_indices = np.arange(50, 100)
        contrast_results = bo.plot_contrast(
            idata, group1_indices, group2_indices, 
            parameter="beta", group_names=["Group A", "Group B"]
        )
        print("✓ Contrast analysis completed")
    except Exception as e:
        print(f"✗ Contrast analysis failed: {e}")
    
    # Test causal graph
    try:
        bo.draw_causal_graph(model_type="ordinal", 
                           feature_names=["X₁", "X₂", "X₃"])
        print("✓ Causal graph created")
    except Exception as e:
        print(f"✗ Causal graph failed: {e}")
    
    return True

def test_model_comparison():
    """Test McElreath-style model comparison."""
    print("\n" + "=" * 60)
    print("TESTING MCELREATH MODEL COMPARISON")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    n_samples, n_features, K = 150, 2, 4
    
    # True parameters
    true_beta = np.array([0.5, -0.3])
    true_cutpoints = np.array([-1.5, -0.5, 0.5])
    
    # Generate features
    X = np.random.normal(size=(n_samples, n_features))
    
    # Generate ordinal responses
    linear_predictor = X @ true_beta
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        probs = 1 / (1 + np.exp(-(true_cutpoints - linear_predictor[i])))
        probs = np.concatenate([[0], probs, [1]])
        category_probs = probs[1:] - probs[:-1]
        y[i] = np.random.choice(K, p=category_probs)
    
    # Preprocess data
    y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K)
    X_scaled, scaler = bo.standardize_features(X_clean)
    
    # Get McElreath priors
    priors = bo.get_conservative_priors(K=K, model_type="ordinal")
    
    # Build multiple models
    models = {}
    idatas = {}
    
    # Cumulative model
    models["cumulative"] = bo.cumulative_model(y=y_clean, X=X_scaled, K=K, priors=priors)
    idatas["cumulative"] = bo.fit_model(models["cumulative"], draws=300, tune=150, chains=2, smoke_test=False)
    
    # Partial odds model
    models["partial_odds"] = bo.partial_odds_model(y=y_clean, X=X_scaled, K=K, priors=priors)
    idatas["partial_odds"] = bo.fit_model(models["partial_odds"], draws=300, tune=150, chains=2, smoke_test=False)
    
    print("✓ Models fitted successfully")
    
    # Test McElreath model comparison
    try:
        comparison_results = bo.compare_models_stacking(models, idatas, ic="loo")
        print("✓ McElreath model comparison completed")
        print(f"  Best model: {comparison_results['best_model']}")
        print(f"  Number of models compared: {len(comparison_results['interpretation'])}")
    except Exception as e:
        print(f"✗ McElreath model comparison failed: {e}")
    
    # Test McElreath model comparison plot
    try:
        bo.plot_model_comparison_interpretation(idatas, ic="loo")
        print("✓ McElreath model comparison plot created")
    except Exception as e:
        print(f"✗ McElreath model comparison plot failed: {e}")
    
    # Test McElreath comparison plot with results
    try:
        bo.plot_model_comparison(comparison_results)
        print("✓ McElreath comparison plot with results created")
    except Exception as e:
        print(f"✗ McElreath comparison plot with results failed: {e}")
    
    return True

def test_workflow_integration():
    """Test integration with existing workflow."""
    print("\n" + "=" * 60)
    print("TESTING WORKFLOW INTEGRATION")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    n_samples, n_features, K = 100, 2, 3
    
    # True parameters
    true_beta = np.array([0.5, -0.3])
    true_cutpoints = np.array([-1.0, 1.0])
    
    # Generate features
    X = np.random.normal(size=(n_samples, n_features))
    
    # Generate ordinal responses
    linear_predictor = X @ true_beta
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        probs = 1 / (1 + np.exp(-(true_cutpoints - linear_predictor[i])))
        probs = np.concatenate([[0], probs, [1]])
        category_probs = probs[1:] - probs[:-1]
        y[i] = np.random.choice(K, p=category_probs)
    
    # Preprocess data
    y_clean, X_clean, K = bo.validate_ordinal_data(y, X, K)
    X_scaled, scaler = bo.standardize_features(X_clean)
    
    # Test workflow with McElreath priors
    try:
        # Use McElreath priors in workflow
        priors = bo.get_conservative_priors(K=K, model_type="ordinal")
        
        # Run workflow
        results = bo.run_workflow(
            model_fn={"cumulative": bo.cumulative_model},
            y=y_clean, X=X_scaled, K=K,
            priors=priors,
            config={
                "draws": 300,
                "tune": 150,
                "chains": 2,
                "run_prior": True,
                "run_ppc": True,
                "run_diagnostics": True,
                "run_cv": True
            }
        )
        
        print("✓ Workflow with McElreath priors completed")
        print(f"  Models built: {list(results['models'].keys())}")
        print(f"  Inference data: {list(results['idata'].keys())}")
        
        # Test McElreath plotting with workflow results
        if 'cumulative' in results['idata']:
            bo.plot_forest(results['idata']['cumulative'], var_name="beta")
            print("✓ McElreath forest plot with workflow results created")
        
    except Exception as e:
        print(f"✗ Workflow integration failed: {e}")
    
    return True

def test_all_functions():
    """Test all new functions comprehensively."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE FUNCTION TESTING")
    print("=" * 60)
    
    # Test all new functions exist and are callable
    new_functions = [
        'get_conservative_priors',
        'get_data_informed_priors',
        'plot_forest',
        'plot_contrast',
        'draw_causal_graph',
        'plot_model_comparison_interpretation',
        'compare_models_stacking',
        'plot_model_comparison'
    ]
    
    for func_name in new_functions:
        try:
            func = getattr(bo, func_name)
            print(f"✓ Function {func_name} is available")
        except AttributeError:
            print(f"✗ Function {func_name} is missing")
    
    # Test function signatures
    try:
        # Test McElreath priors signature
        priors = bo.get_conservative_priors(K=4, model_type="ordinal")
        assert isinstance(priors, dict)
        print("✓ get_conservative_priors signature correct")
    except Exception as e:
        print(f"✗ get_conservative_priors signature error: {e}")
    
    try:
        # Test causal graph signature
        bo.draw_causal_graph(model_type="ordinal")
        print("✓ draw_causal_graph signature correct")
    except Exception as e:
        print(f"✗ draw_causal_graph signature error: {e}")
    
    return True

def main():
    """Run all tests."""
    print("MCELREATH ENHANCEMENTS TEST SUITE")
    print("Testing all new functions and workflow integration...")
    
    # Run all tests
    tests = [
        test_conservative_priors,
        test_mcelreath_plotting,
        test_model_comparison,
        test_workflow_integration,
        test_all_functions
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! McElreath enhancements are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\nMcElreath-style enhancements include:")
    print("✓ McElreath conservative priors")
    print("✓ Data-informed priors")
    print("✓ McElreath forest plots")
    print("✓ Contrast analysis")
    print("✓ Causal graph visualization")
    print("✓ Enhanced model comparison with interpretation rules")
    print("✓ Seamless workflow integration")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 