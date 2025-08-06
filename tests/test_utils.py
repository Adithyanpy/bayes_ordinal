"""
Tests for utility functions.
"""

import numpy as np
import pytest
import pandas as pd
from bayes_ordinal.utils import (
    validate_ordinal_data,
    encode_categorical_features,
    standardize_features,
    create_group_indices,
    compute_category_proportions
)

class TestUtils:
    """Test utility functions."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n, self.K = 100, 4
        self.X = np.random.normal(size=(self.n, 3))
        self.y = np.random.randint(0, self.K, size=self.n)
    
    def test_validate_ordinal_data_valid(self):
        """Test data validation with valid data."""
        y_clean, X_clean, K = validate_ordinal_data(self.y, self.X, self.K)
        
        assert np.array_equal(y_clean, self.y)
        assert np.array_equal(X_clean, self.X)
        assert K == self.K
    
    def test_validate_ordinal_data_infer_k(self):
        """Test data validation with K inference."""
        y_clean, X_clean, K = validate_ordinal_data(self.y, self.X)
        
        assert K == self.K
        assert np.array_equal(y_clean, self.y)
    
    def test_validate_ordinal_data_shape_mismatch(self):
        """Test data validation with shape mismatch."""
        X_wrong = np.random.normal(size=(self.n + 1, 3))
        
        with pytest.raises(ValueError, match="samples"):
            validate_ordinal_data(self.y, X_wrong, self.K)
    
    def test_validate_ordinal_data_missing_values(self):
        """Test data validation with missing values."""
        y_with_nan = self.y.astype(float).copy()
        y_with_nan[0] = np.nan
        
        with pytest.raises(ValueError, match="missing values"):
            validate_ordinal_data(y_with_nan, self.X, self.K)
    
    def test_validate_ordinal_data_negative_values(self):
        """Test data validation with negative values."""
        y_negative = self.y.copy()
        y_negative[0] = -1
        
        with pytest.raises(ValueError, match="negative values"):
            validate_ordinal_data(y_negative, self.X, self.K)
    
    def test_validate_ordinal_data_gaps(self):
        """Test data validation with gaps in categories."""
        y_gaps = np.array([0, 0, 2, 2, 3, 3])  # Missing category 1
        
        with pytest.raises(ValueError, match="gaps"):
            validate_ordinal_data(y_gaps, self.X[:6], K=4)
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        # Create DataFrame with categorical column
        df = pd.DataFrame({
            'numeric': np.random.normal(size=self.n),
            'categorical': np.random.choice(['A', 'B', 'C'], size=self.n)
        })
        
        X_encoded, encoders = encode_categorical_features(df, ['categorical'])
        
        assert X_encoded.shape == (self.n, 2)
        assert 'categorical' in encoders
        assert hasattr(encoders['categorical'], 'fit_transform')
    
    def test_standardize_features(self):
        """Test feature standardization."""
        X_scaled, scaler = standardize_features(self.X)
        
        assert X_scaled.shape == self.X.shape
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_standardize_features_with_scaler(self):
        """Test feature standardization with pre-fitted scaler."""
        X_scaled1, scaler = standardize_features(self.X)
        X_scaled2, _ = standardize_features(self.X, scaler)
        
        assert np.allclose(X_scaled1, X_scaled2)
    
    def test_create_group_indices(self):
        """Test group index creation."""
        groups = np.random.choice(['group1', 'group2', 'group3'], size=self.n)
        
        group_idx, n_groups, group_map = create_group_indices(groups)
        
        assert len(group_idx) == self.n
        assert n_groups == 3
        assert len(group_map) == 3
        assert all(idx in group_map.values() for idx in group_idx)
    
    def test_compute_category_proportions(self):
        """Test category proportion computation."""
        proportions = compute_category_proportions(self.y, self.K)
        
        assert len(proportions) == self.K
        assert np.allclose(proportions.sum(), 1.0)
        assert all(p >= 0 for p in proportions) 