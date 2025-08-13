import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from ..plot_utils import show_and_save

def run_posterior_predictive(
    model,
    idata: az.InferenceData,
    var_name: str = "y",  # PyMCOrdinal docs use "y" as default
    kind: str = "proportions",
    figsize=(6, 4),
):
    """
    Run posterior predictive checks and visualize discrete ordinal outcomes.

    Supports visualization modes:
      - "proportions": stacked bar overlays of sampled vs observed proportions
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
    kind : {"proportions", "hist", "ecdf"}, default="proportions"
        Type of plot to generate.
    figsize : tuple, default=(6, 4)
        Size of the output figure.

    Returns
    -------
    ppc : az.InferenceData
        Posterior predictive samples as ArviZ InferenceData.

    Examples
    --------
    >>> ppc = run_posterior_predictive(model, idata, kind="proportions")
    >>> ppc.posterior_predictive["y"].shape
    """
    # 1) sample posterior predictive
    ppc = pm.sample_posterior_predictive(
        idata,
        model=model,
        var_names=[var_name],
        return_inferencedata=True,
    )

    if kind == "proportions":
        # observed proportions - handle variable name variations
        try:
            y_obs = idata.observed_data[var_name].values
        except KeyError:
            # Try to find the variable with the right name pattern
            available_vars = list(idata.observed_data.data_vars.keys())
            matching_vars = [v for v in available_vars if var_name in v]
            if matching_vars:
                y_obs = idata.observed_data[matching_vars[0]].values
                print(f"Using observed variable: {matching_vars[0]}")
            else:
                print(f"Warning: Could not find observed variable '{var_name}'")
                print(f"Available variables: {available_vars}")
                return ppc
        
        K = int(y_obs.max()) + 1
        counts_obs = np.bincount(y_obs, minlength=K)
        prop_obs = counts_obs / counts_obs.sum()

        # posterior predictive draws - handle variable name variations
        try:
            arr = ppc.posterior_predictive[var_name].stack(
                samples=("chain", "draw")
            ).values  # shape = (n_samples, N)
        except KeyError:
            # Try to find the variable with the right name pattern
            available_vars = list(ppc.posterior_predictive.data_vars.keys())
            matching_vars = [v for v in available_vars if var_name in v]
            if matching_vars:
                arr = ppc.posterior_predictive[matching_vars[0]].stack(
                    samples=("chain", "draw")
                ).values
                print(f"Using predictive variable: {matching_vars[0]}")
            else:
                print(f"Warning: Could not find predictive variable '{var_name}'")
                print(f"Available variables: {available_vars}")
                return ppc
        
        # compute proportions per sample
        props = np.stack(
            [np.bincount(d, minlength=K) / d.size for d in arr],
            axis=0,
        )  # (n_samples, K)

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        # draw‐by-draw faint bars
        for p in props:
            ax.bar(np.arange(K), p, width=0.8, alpha=0.05, color="C0")
        # posterior predictive **mean** bar
        ax.bar(
            np.arange(K),
            props.mean(axis=0),
            width=0.8,
            alpha=0.5,
            color="C0",
            label="posterior predictive",
        )
        # observed outline
        ax.step(
            np.arange(K),
            prop_obs,
            where="mid",
            lw=2,
            color="C1",
            label="observed",
        )
        ax.set_xlabel("Ordinal Category")
        ax.set_ylabel("Proportion")
        ax.set_title("Posterior Predictive Check")
        ax.legend()
        plt.grid(True, alpha=0.3)
        show_and_save("posterior_predictive_proportions")

    elif kind in ("hist", "ecdf"):
        combined = idata.copy()
        combined.extend(ppc)
        plot_kind = "hist" if kind == "hist" else "cumulative"
        if kind == "hist":
            # for discrete variables, kind="kde" will produce a histogram
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
        raise ValueError("kind must be one of 'proportions','hist','ecdf'")

    return ppc
