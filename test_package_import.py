#!/usr/bin/env python3
"""
Simple test script to verify package imports and basic functionality.
"""

import sys
import numpy as np

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing package imports...")
    
    try:
        import bayes_ordinal as bo
        print("✓ Main package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import main package: {e}")
        return False
    
    try:
        from bayes_ordinal.models.cumulative import cumulative_model
        print("✓ Cumulative model imported")
    except ImportError as e:
        print(f"✗ Failed to import cumulative model: {e}")
        return False
    
    try:
        from bayes_ordinal.models.partial_odds import partial_odds_model
        print("✓ Partial odds model imported")
    except ImportError as e:
        print(f"✗ Failed to import partial odds model: {e}")
        return False
    
    try:
        from bayes_ordinal.priors import get_default_priors
        print("✓ Priors module imported")
    except ImportError as e:
        print(f"✗ Failed to import priors: {e}")
        return False
    
    try:
        from bayes_ordinal.workflow.fitting import fit_model
        print("✓ Workflow fitting imported")
    except ImportError as e:
        print(f"✗ Failed to import workflow fitting: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without running MCMC."""
    print("\nTesting basic functionality...")
    
    try:
        import bayes_ordinal as bo
        
        # Generate test data
        np.random.seed(42)
        n, K = 20, 4
        X = np.random.normal(size=(n, 2))
        y = np.random.randint(0, K, size=n)
        
        # Test data validation
        y_clean, X_clean, K_clean = bo.validate_ordinal_data(y, X, K)
        print("✓ Data validation works")
        
        # Test prior creation
        priors = bo.get_default_priors()
        print("✓ Prior creation works")
        
        # Test model creation (without sampling)
        model = bo.cumulative_model(y=y, X=X, K=K, priors=priors)
        print("✓ Model creation works")
        
        # Test partial odds model
        partial_model = bo.partial_odds_model(y=y, X=X, K=K, priors=priors)
        print("✓ Partial odds model creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from bayes_ordinal.config import get_default_config, get_production_config
        
        config = get_default_config()
        print("✓ Default configuration works")
        
        prod_config = get_production_config()
        print("✓ Production configuration works")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Bayesian Ordinal Regression Package Test")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test configuration
    if not test_configuration():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! Package is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 