"""
Tests for cumulative ordinal regression model.
"""

import numpy as np
import pytest
import pymc as pm
from bayes_ordinal.models.cumulative import cumulative_model
from bayes_ordinal.priors import get_default_priors, validate_priors

class TestCumulativeModel:
    """Test cumulative ordinal regression model."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n, self.K = 50, 4
        self.X = np.random.normal(size=(self.n, 2))
        self.y = np.random.randint(0, self.K, size=self.n)
        self.priors = get_default_priors()
    
    def test_model_creation(self):
        """Test that model can be created."""
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            priors=self.priors
        )
        assert isinstance(model, pm.Model)
    
    def test_model_with_different_links(self):
        """Test model with different link functions."""
        links = ["logit", "probit", "cloglog"]
        
        for link in links:
            model = cumulative_model(
                y=self.y, X=self.X, K=self.K,
                link=link,
                priors=self.priors
            )
            assert isinstance(model, pm.Model)
    
    def test_model_with_hierarchical(self):
        """Test model with hierarchical structure."""
        group_idx = np.random.randint(0, 3, size=self.n)
        n_groups = 3
        
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            group_idx=group_idx,
            n_groups=n_groups,
            priors=self.priors
        )
        assert isinstance(model, pm.Model)
    
    def test_model_variables(self):
        """Test that model has expected variables."""
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            priors=self.priors
        )
        
        # Check for expected variables (PyMC adds model name prefix)
        var_names = [var.name for var in model.free_RVs]
        assert any("alpha" in name for name in var_names)
        assert any("beta" in name for name in var_names)
        
        # Check observed variable (PyMC adds model name prefix)
        obs_names = [var.name for var in model.observed_RVs]
        assert any("y_obs" in name for name in obs_names)
    
    def test_invalid_link(self):
        """Test that invalid link raises error."""
        with pytest.raises(ValueError, match="Unknown link"):
            cumulative_model(
                y=self.y, X=self.X, K=self.K,
                link="invalid_link"
            )
    
    def test_constrained_uniform_cutpoints(self):
        """Test constrained uniform cutpoints option."""
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            constrained_uniform=True,
            priors=self.priors
        )
        assert isinstance(model, pm.Model)
        
        # Check that we have the expected variables (PyMC adds model name prefix)
        var_names = [var.name for var in model.free_RVs]
        assert any("alpha" in name or "cuts_unknown" in name for name in var_names)
        assert any("beta" in name for name in var_names)
    
    def test_default_initvals(self):
        """Test that model has default initial values."""
        model = cumulative_model(
            y=self.y, X=self.X, K=self.K,
            priors=self.priors
        )
        
        assert hasattr(model, "_default_initvals")
        # Check for alpha or cuts_unknown in initvals
        has_alpha = any("alpha" in key for key in model._default_initvals.keys())
        assert has_alpha or any("cuts_unknown" in key for key in model._default_initvals.keys())
        assert any("beta" in key for key in model._default_initvals.keys())
        
        # Check lengths (find the actual keys)
        alpha_key = next((key for key in model._default_initvals.keys() if "alpha" in key or "cuts_unknown" in key), None)
        beta_key = next((key for key in model._default_initvals.keys() if "beta" in key), None)
        
        if alpha_key:
            assert len(model._default_initvals[alpha_key]) == self.K - 1
        if beta_key:
            assert len(model._default_initvals[beta_key]) == self.X.shape[1]
