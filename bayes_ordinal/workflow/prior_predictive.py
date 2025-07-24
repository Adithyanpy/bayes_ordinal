import arviz as az
import matplotlib.pyplot as plt
import pymc as pm

def run_prior_predictive(
    model: pm.Model,
    draws: int = 200,
    plot: bool = True
) -> az.InferenceData:
    """
    Run prior predictive checks on a PyMC model.

    Parameters
    ----------
    model : pm.Model
        The un-sampled model (with priors only).
    draws : int
        Number of prior predictive samples.
    plot : bool
        Whether to plot category counts from the prior predictive.

    Returns
    -------
    idata : arviz.InferenceData
        The prior predictive samples.
    """
    with model:
        idata = pm.sample_prior_predictive(draws=draws, return_inferencedata=True)

    if plot:
        # Assume the observed variable is named "y_obs"
        y_pp = idata.prior_predictive["y_obs"]
        # Plot the distribution of category counts
        az.plot_ppc(idata, var_names=["y_obs"], group="prior", kind="hist", show=True)
        plt.title("Prior Predictive Distribution of y_obs")
        plt.xlabel("Category")
        plt.ylabel("Frequency")

    return idata
