
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
from arviz.stats import hdi
from typing import Callable, Sequence

# Import your fit_ordinal_model helper
from .fitting import fit_ordinal_model

def prior_sensitivity(
    build_model: Callable[..., pm.Model],
    y: np.ndarray,
    X: np.ndarray,
    K: int,
    hyper_name: str,
    hyper_values: Sequence[float],
    trace_var: str,
    draws: int = 300,
    tune: int = 300,
    chains: int = 4,
    hdi_prob: float = 0.9,
):
    """
    Sweep a single prior hyperparameter and examine its effect on posterior.

    Useful for checking robustness of posterior inferences to prior choices.

    Parameters
    ----------
    build_model : Callable
        Function that returns a PyMC model when called with (y, X, K, priors).
    y : np.ndarray
        Target variable (ordinal categories).
    X : np.ndarray
        Feature matrix.
    K : int
        Number of outcome categories.
    hyper_name : str
        Name of the hyperparameter to sweep (e.g., "sigma").
    hyper_values : list of float
        Values to try for the hyperparameter.
    trace_var : str
        Name of the variable to extract from posterior.
    draws : int, default=300
        Number of posterior draws.
    tune : int, default=300
        Number of tuning steps.
    chains : int, default=4
        Number of chains.
    hdi_prob : float, default=0.9
        Width of HDI interval.

    Returns
    -------
    None
        Displays a plot showing sensitivity analysis results

    Examples
    --------
    >>> prior_sensitivity(cumulative_model, y, X, 4, "sigma", [1, 3, 5], "cutpoints")
    """
    means, lowers, uppers = [], [], []

    for val in hyper_values:
        priors = {hyper_name: val}
        model = build_model(y, X, K, priors=priors)

        #  use fit_ordinal_model instead of direct pm.sample 
        idata = fit_ordinal_model(
            model,
            draws=draws,
            tune=tune,
            chains=chains,
            return_inferencedata=True,
            progressbar=False           # passed through to pm.sample
        )

        # flatten all samples of trace_var
        arr = idata.posterior[trace_var].values.flatten()
        means.append(arr.mean())
        low, high = hdi(arr, hdi_prob=hdi_prob)
        lowers.append(low)
        uppers.append(high)

    # Plot mean +/- HDI
    plt.figure(figsize=(6, 4))
    yerr = [
        np.array(means) - np.array(lowers),
        np.array(uppers) - np.array(means)
    ]
    plt.errorbar(
        hyper_values,
        means,
        yerr=yerr,
        fmt="o-",
        capsize=5,
        lw=2,
    )
    plt.xlabel(f"Prior `{hyper_name}`")
    plt.ylabel(f"Posterior mean +/- {int(hdi_prob*100)}% HDI of `{trace_var}`")
    plt.title(f"Sensitivity of `{trace_var}` to prior `{hyper_name}`")
    plt.tight_layout()
    plt.show()


def plot_influential(
    idata: az.InferenceData,
    threshold: float = 0.7,
    use_az_khat: bool = True
):
    """
    Plot influence diagnostics based on Pareto-k values from LOO.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior samples.
    threshold : float, default=0.7
        Value above which k is flagged as problematic.
    use_az_khat : bool, default=True
        If True, use ArviZ's built-in `plot_khat`. Else plot a histogram.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_influential(idata, threshold=0.7)
    """
    loo = az.loo(idata, pointwise=True)

    if use_az_khat:
        az.plot_khat(loo, threshold=threshold)
        plt.tight_layout()
        plt.show()
        return

    ks = np.asarray(loo.pareto_k.values).ravel()
    plt.figure(figsize=(6, 3))
    plt.hist(
        ks,
        bins=20,
        range=(0, 1),
        alpha=0.7,
        edgecolor="k",
        rwidth=0.8
    )
    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"k = {threshold}"
    )
    plt.xlabel("Pareto k")
    plt.ylabel("Count")
    plt.title("Influence Diagnostics (Pareto-k)")
    plt.legend()
    plt.tight_layout()
    plt.show()
