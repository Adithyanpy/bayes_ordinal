from typing import Callable, Union
import numpy as np
import pymc as pm
import arviz as az

from .prior_predictive import run_prior_predictive
from .fitting import fit_model
from .posterior_predictive import run_posterior_predictive
from .diagnostics import summarize_diagnostics
from .cross_validation import compare_models


def run_workflow(
    model_fn: Union[
        Callable[..., pm.Model],
        dict[str, Callable[..., pm.Model]]
    ],
    y: np.ndarray,
    X: np.ndarray,
    K: int,
    priors: dict | None = None,
    group_idx: np.ndarray | None = None,
    n_groups: int | None = None,
    config: dict | None = None
) -> dict:
    """
    Run full Bayesian ordinal regression workflow from prior to diagnostics.

    Parameters
    ----------
    model_fn : callable or dict
        Either a single model constructor (e.g. `cumulative_model`) or a dict mapping
        names to constructors (e.g. {"cum": cumulative_model, "partial": partial_odds_model}).
    y : array-like
        Ordinal outcome vector.
    X : array-like
        Covariates (n_samples, n_features).
    K : int
        Number of ordinal categories.
    priors : dict, optional
        Custom prior overrides.
    group_idx : array-like, optional
        Index for varying intercepts.
    n_groups : int, optional
        Number of unique groups.
    config : dict, optional
        Workflow config. Keys include:
            - draws, tune, chains, smoke_test
            - run_prior, run_ppc, run_diagnostics, run_cv

    Returns
    -------
    results : dict
        Dictionary with components for each model and step.
    """
    config = config or {}
    results: dict = {}

    # ─── Build one or more models based on model_fn ─────────────────────────────────
    if isinstance(model_fn, dict):
        models = {
            name: fn(y=y, X=X, K=K, priors=priors, group_idx=group_idx, n_groups=n_groups)
            for name, fn in model_fn.items()
        }
    else:
        models = {"model": model_fn(y=y, X=X, K=K, priors=priors, group_idx=group_idx, n_groups=n_groups)}
    results["models"] = models

    # ─── Loop over each model through the workflow ──────────────────────────────────
    for name, model in models.items():
        # 1) Prior predictive check
        if config.get("run_prior", True):
            prior_idata = run_prior_predictive(model, draws=config.get("prior_draws", 200))
            results.setdefault("prior_idata", {})[name] = prior_idata

        # 2) Fit model
        idata = fit_model(
            model,
            draws=config.get("draws", 1000),
            tune=config.get("tune", 1000),
            chains=config.get("chains", 4),
            smoke_test=config.get("smoke_test", False),
            return_inferencedata=True,
            progressbar=config.get("progressbar", True),
        )
        results.setdefault("idata", {})[name] = idata

        # 3) Posterior predictive check
        if config.get("run_ppc", True):
            ppc = run_posterior_predictive(
                model,
                idata,
                kind=config.get("ppc_kind", "proportions")
            )
            # if you only built one model, just stash the raw idata:
            if len(models) == 1:
                results["ppc"] = ppc
            else:
                results.setdefault("ppc", {})[name] = ppc
 

        # 4) Diagnostics
        if config.get("run_diagnostics", True):
            diag_df = summarize_diagnostics(idata)
            if len(models) == 1:
                results["diagnostics"] = diag_df
            else:
                results.setdefault("diagnostics", {})[name] = diag_df

        # 5) Cross‐validation
        if config.get("run_cv", True):
            cv_metric = config.get("cv_metric", "loo")
            # multi‐model stacking/compare
            if isinstance(model_fn, dict) and len(models) > 1:
                comps = compare_models(
                    {n: m for n, m in results["models"].items()},
                    {n: i for n, i in results["idata"].items()},
                    ic=cv_metric,
                )
                results["cv"] = comps
            else:
                # single‐model: just compute loo or waic on its InferenceData
                if cv_metric == "loo":
                    results["cv"] = az.loo(idata)
                else:
                    results["cv"] = az.waic(idata)

    return results
