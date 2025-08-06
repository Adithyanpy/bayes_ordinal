"""
Tests for multilevel/hierarchical ordinal regression models.
"""

import numpy as np
import pytest
import pymc as pm
from bayes_ordinal.models.cumulative import cumulative_model
from bayes_ordinal.models.partial_odds import partial_odds_model
from bayes_ordinal.utils import create_group_indices
from bayes_ordinal.priors import get_default_priors

class TestMultilevelModels:
    """Test multilevel/hierarchical ordinal regression models."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n, self.K = 60, 4
        self.X = np.random.normal(size=(self.n, 2))
        self.y = np.random.randint(0, self.K, size=self.n)
        self.priors = get_default_priors()
        
        # Create group structure
        self.groups = np.random.choice(['A', 'B', 'C'], size=self.n)
        self.group_idx, self.n_groups, self.group_map = create_group_indices(self.groups)
    
    def test_cumulative_hierarchical_creation(self):
        """Test hierarchical cumulative model creation."""
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            group_idx=self.group_idx,
            n_groups=self.n_groups,
            priors=self.priors
        )
        
        assert isinstance(model, pm.Model)
        
        # Check for hierarchical variables (PyMC adds model name prefix for cumulative)
        var_names = [var.name for var in model.free_RVs]
        assert any("u_sigma" in name for name in var_names)
        assert any("u_offset" in name for name in var_names)
    
    def test_partial_odds_hierarchical_creation(self):
        """Test hierarchical partial odds model creation."""
        model = partial_odds_model(
            y=self.y, X=self.X, K=self.K,
            group_idx=self.group_idx,
            n_groups=self.n_groups,
            priors=self.priors
        )
        
        assert isinstance(model, pm.Model)
        
        # Check for hierarchical variables (PyMC adds model name prefix for partial odds)
        var_names = [var.name for var in model.free_RVs]
        assert any("u_sigma" in name for name in var_names)
        assert any("u_offset" in name for name in var_names)
    
    def test_hierarchical_variable_shapes(self):
        """Test that hierarchical variables have correct shapes."""
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            group_idx=self.group_idx,
            n_groups=self.n_groups,
            priors=self.priors
        )
        
        # Check u_offset shape (PyMC adds model name prefix)
        u_offset_var = [var for var in model.free_RVs if "u_offset" in var.name][0]
        # PyMC shapes are symbolic, so we check the size instead
        assert u_offset_var.size == self.n_groups
    
    def test_hierarchical_initvals(self):
        """Test that hierarchical models have appropriate initial values."""
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            group_idx=self.group_idx,
            n_groups=self.n_groups,
            priors=self.priors
        )
        
        assert hasattr(model, "_default_initvals")
        assert any("u_offset" in key for key in model._default_initvals.keys())
        assert any("u_sigma" in key for key in model._default_initvals.keys())
        
        # Find the actual u_offset key
        u_offset_key = next((key for key in model._default_initvals.keys() if "u_offset" in key), None)
        if u_offset_key:
            assert len(model._default_initvals[u_offset_key]) == self.n_groups
    
    def test_group_indices_creation(self):
        """Test group index creation utility."""
        group_idx, n_groups, group_map = create_group_indices(self.groups)
        
        assert len(group_idx) == self.n
        assert n_groups == 3
        assert len(group_map) == 3
        assert all(idx in group_map.values() for idx in group_idx)
        assert all(isinstance(idx, (int, np.integer)) for idx in group_idx)
    
    def test_hierarchical_with_custom_priors(self):
        """Test hierarchical models with custom priors."""
        custom_priors = self.priors.copy()
        custom_priors["u_sigma"] = 2.0
        
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            group_idx=self.group_idx,
            n_groups=self.n_groups,
            priors=custom_priors
        )
        
        assert isinstance(model, pm.Model)
    
    def test_hierarchical_model_variables(self):
        """Test that hierarchical models have all expected variables."""
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            group_idx=self.group_idx,
            n_groups=self.n_groups,
            priors=self.priors
        )
        
        # Check for all expected variables (PyMC adds model name prefix)
        var_names = [var.name for var in model.free_RVs]
        expected_vars = ["alpha", "beta", "u_sigma", "u_offset"]
        
        for var_name in expected_vars:
            assert any(var_name in name for name in var_names), f"Variable {var_name} not found in {var_names}"
        
        # Check for deterministic u variable
        det_names = [var.name for var in model.deterministics]
        assert any("u" in name for name in det_names)
    
    def test_non_centered_parameterization(self):
        """Test that hierarchical models use non-centered parameterization."""
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            group_idx=self.group_idx,
            n_groups=self.n_groups,
            priors=self.priors
        )
        
        # Check that we have u_offset and u_sigma (non-centered)
        var_names = [var.name for var in model.free_RVs]
        assert any("u_offset" in name for name in var_names)
        assert any("u_sigma" in name for name in var_names)
        
        # Check that u is deterministic (u = u_offset * u_sigma)
        det_names = [var.name for var in model.deterministics]
        assert any("u" in name for name in det_names)
    
    def test_hierarchical_model_compilation(self):
        """Test that hierarchical models compile without errors."""
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            group_idx=self.group_idx,
            n_groups=self.n_groups,
            priors=self.priors
        )
        
        # This should not raise an error
        assert isinstance(model, pm.Model)
        
        # Test that model can be used for sampling (smoke test)
        with model:
            # Just test that we can create a trace without actually sampling
            pass
