import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def run_posterior_predictive(
    model,
    idata: az.InferenceData,
    var_name: str = "y",  
    kind: str = "hist",
    figsize=(6, 4),
):
    """
    Run posterior predictive checks and visualize discrete ordinal outcomes.

    Supports visualization modes:
      - "hist": histogram-based plot (via ArviZ)
      - "ecdf": empirical cumulative distribution function (via ArviZ)

    Parameters
    ----------
    model : pm.Model
        The PyMC model used for sampling.
    idata : az.InferenceData
        Inference data containing the posterior draws.
    var_name : str, default="y"
        Name of the observed variable (PyMCOrdinal convention: "y").
    kind : {"hist", "ecdf"}, default="hist"
        Type of plot to generate.
    figsize : tuple, default=(6, 4)
        Size of the output figure.

    Returns
    -------
    ppc : az.InferenceData
        Posterior predictive samples as ArviZ InferenceData.

    Examples
    --------
    >>> ppc = run_posterior_predictive(model, idata, kind="hist")
    >>> ppc.posterior_predictive["y"].shape
    """
    # 1) sample posterior predictive
    ppc = pm.sample_posterior_predictive(
        idata,
        model=model,
        var_names=[var_name],
        return_inferencedata=True,
    )

    if kind in ("hist", "ecdf"):
        combined = idata.copy()
        combined.extend(ppc)
        plot_kind = "hist" if kind == "hist" else "cumulative"
        if kind == "hist":
            az.plot_ppc(
                combined,
                var_names=[var_name],
                kind="kde",       # use 'kde' so discrete data get plotted as histograms
                figsize=figsize,
                show=True
            )
        else:
            # ECDF view
            az.plot_ppc(
                combined,
                var_names=[var_name],
                kind="cumulative",
                figsize=figsize,
                show=True
            )

    else:
        raise ValueError("kind must be one of 'hist','ecdf'")

    return ppc
