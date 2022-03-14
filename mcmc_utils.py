import emcee
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def setup_params(param_dict):

    param_name = [
        key for key in param_dict.keys() if not param_dict[key]["fixed"]
    ]
    fixed_param_name = [
        key for key in param_dict.keys() if param_dict[key]["fixed"]
    ]
    bound = [
        param_dict[key]["bound"]
        for key in param_dict.keys()
        if not param_dict[key]["fixed"]
    ]
    initial_state = [
        param_dict[key]["p0"]
        for key in param_dict.keys()
        if not param_dict[key]["fixed"]
    ]

    return param_name, fixed_param_name, bound, initial_state


def condition(p, b):
    if (p >= b[0]) and (p <= b[1]):
        return True
    return False


def log_prior(param, bound):
    for p, b in zip(param, bound):
        if not condition(p, b):
            return -np.inf
    return 0.0


def run_emcee(
    log_probability,
    initial_state,
    args=None,
    nwalker=200,
    nstep=500,
    initial_blob_mag=1e-4,
    pool=None,
    progress=True,
    blobs_dtype=None,
    nthreads=None
):

    # set dimension and initial guesses
    ndim = len(initial_state)
    p0 = initial_state + initial_blob_mag * np.random.randn(nwalker, ndim)

    # set smapler
    #with ProcessPool(node=nthread) as pool:
    #with multiprocessing.Pool(processes=nthread) as pool:
    #with Pool(nthread) as pool:

    if pool is not None:
        sampler = emcee.EnsembleSampler(
            nwalker, ndim, log_probability, args=args, pool=pool, blobs_dtype=blobs_dtype,
        )

        # run
        print(
            "starting to run the MCMC sampling using {} threads with: \n \t initial state:".format(nthreads),
            initial_state,
            "\n \t number of walkers:",
            nwalker,
            "\n \t number of steps:",
            nstep
        )
        sampler.run_mcmc(p0, nstep, progress=progress)

    elif nthreads is None or nthreads == 1:
        sampler = emcee.EnsembleSampler(
            nwalker, ndim, log_probability, args=args, blobs_dtype=blobs_dtype,
        )

        # run
        print(
            "starting to run the MCMC sampling in a single thread with: \n \t initial state:",
            initial_state,
            "\n \t number of walkers:",
            nwalker,
            "\n \t number of steps:",
            nstep
        )
        sampler.run_mcmc(p0, nstep, progress=progress)

    else:
        with multiprocessing.get_context("spawn").Pool(processes=nthreads) as pool:
            sampler = emcee.EnsembleSampler(
                nwalker, ndim, log_probability, args=args, pool=pool, blobs_dtype=blobs_dtype,
            )

            # run
            print(
                "starting to run the MCMC sampling using {} threads with: \n \t initial state:".format(nthreads),
                initial_state,
                "\n \t number of walkers:",
                nwalker,
                "\n \t number of steps:",
                nstep
            )
            sampler.run_mcmc(p0, nstep, progress=progress)

    return sampler


def plot_walker(sample, nburnin=0, labels=None, histogram=True):
    #     # Check the length of the label list.

    # 	if labels is not None:
    # 		if sample.shape[0] != len(labels):
    # 			raise ValueError("Incorrect number of labels.")

    sample = sample.transpose((2, 0, 1))
    # Cycle through the plots.

    figset = []

    for i, s in enumerate(sample):
        fig, ax = plt.subplots()
        for walker in s.T:
            ax.plot(walker, alpha=0.1, color="k")
        ax.set_xlabel("Step number")
        if labels is not None:
            ax.set_ylabel(labels[i])
        ax.axvline(nburnin, ls="dotted", color="tab:blue")
        ax.set_xlim(0, s.shape[0])

        # Include the histogram.

        if histogram:
            fig.set_size_inches(
                1.37 * fig.get_figwidth(), fig.get_figheight(), forward=True
            )
            ax_divider = make_axes_locatable(ax)
            bins = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
            hist, _ = np.histogram(
                s[nburnin :].flatten(), bins=bins, density=True
            )
            bins = np.average([bins[1:], bins[:-1]], axis=0)
            ax1 = ax_divider.append_axes("right", size="35%", pad="2%")
            ax1.fill_betweenx(
                bins,
                hist,
                np.zeros(bins.size),
                step="mid",
                color="darkgray",
                lw=0.0,
            )
            ax1.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
            ax1.set_xlim(0, ax1.get_xlim()[1])
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            ax1.tick_params(which="both", left=0, bottom=0, top=0, right=0)
            ax1.spines["right"].set_visible(False)
            ax1.spines["bottom"].set_visible(False)
            ax1.spines["top"].set_visible(False)

            # get percentile
            q = np.percentile(s[nburnin :].flatten(), [16, 50, 84])
            for val in q:
                ax1.axhline(val, ls="dashed", color="black")
            text = (
                labels[i]
                + r"$ = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
                    q[1], np.diff(q)[0], np.diff(q)[1]
                )
                if labels is not None
                else r"${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
                    q[1], np.diff(q)[1], np.diff(q)[0]
                )
            )
            ax1.text(0.5, 1.0, text, transform=ax1.transAxes, ha="center", va="top")

        figset.append(fig)

    return figset