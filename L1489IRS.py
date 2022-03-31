import matplotlib.pyplot as plt
import numpy as np
from galario.double import sampleProfile
from fileio import import_ms, export_ms
from uvanalysis_utils import set_grid, plot_uv, deproject_vis, bin_vis
from mcmc_utils import setup_params, log_prior, run_emcee, plot_walker
from imageanalysis_utils import Gaussian1d, GaussianRing1d
import astropy.constants as ac
import astropy.units as units
import pickle
import corner
import multiprocessing

c = ac.c.to(units.m / units.s).value
pi = np.pi
deg = pi / 180.0  # in rad
arcsec = pi / 180.0 / 3600.0  # in rad

geometrical_param = ["PA", "incl", "dRA", "dDec"]
gp_default = {
    "PA": {"p0": 0.0, "bound": (0.0, 180.0), "fixed": True},
    "incl": {"p0": 0.0, "bound": (0.0, 90.0), "fixed": True},
    "dRA": {"p0": 0.0, "bound": (-5.0, 5.0), "fixed": True},
    "dDec": {"p0": 0.0, "bound": (-5.0, 5.0), "fixed": True},
}
accepted_chain_length = 100


################## Target name ###################
source = "L1489IRS"
################## Define the model function and parameters ##################
# model function on the image plane; modify as needed
# def model_func_1d(r, I_g, sigma_g, I_p, sigma_p):
#     return 10**I_g * np.exp(-0.5 * r**2 / sigma_g**2) + 10**I_p * np.exp(
#         -0.5 * r**2 / sigma_p**2
#     )


# def model_func_1d(r, I_p, sigma_p, I_r, r_r, sigma_r):
#     return 10**I_p * np.exp(-0.5 * r**2 / sigma_p**2) + 10**I_r * np.exp(
#         -0.5 * (r - r_r) ** 4 / sigma_r**4
#     )


# model function derived from image fit
model_name = "PointSource_GaussianRing_Gaussian_longrun"


def model_func_1d(r, F_c, sigma_c, r0_r, F_r, sigma_r, F_b, sigma_b):
    return (
        Gaussian1d(r, F_c, sigma_c)
        + GaussianRing1d(r, r0_r, F_r, sigma_r)
        + Gaussian1d(r, F_b, sigma_b)
    )


# dictionary of parameters including initial guess, bound, and fixed/free
param_dict = {
    # "I_g": {"p0": 8.5, "bound": (-2.0, 20.0), "fixed": False},
    # "sigma_g": {"p0": 2.0, "bound": (0.1, 10), "fixed": False},
    "F_c": {"p0": 10.59, "bound": (5, 15), "fixed": False},
    "sigma_c": {"p0": 0.01, "bound": (1e-5, 0.2), "fixed": False},
    "r0_r": {"p0": 0.47, "bound": (0.01, 1.5), "fixed": False},
    "F_r": {"p0": 8.31, "bound": (3, 13), "fixed": False},
    "sigma_r": {"p0": 0.20, "bound": (1e-2, 1.5), "fixed": False},
    "F_b": {"p0": 9.14, "bound": (4, 14), "fixed": False},
    "sigma_b": {"p0": 1.59, "bound": (0.3, 5), "fixed": False},
    "PA": {"p0": 68.9, "bound": (0, 180), "fixed": False},
    "incl": {"p0": 72.86, "bound": (0, 90), "fixed": False},
    "dRA": {"p0": 0.0, "bound": (-2, 2), "fixed": False},
    "dDec": {"p0": 0.0, "bound": (-2, 2), "fixed": False},
}
###############################################################################


########## load the visibility and define various functions for MCMC ##########
# load the visibility
print("Loading visibility data...")
datafilepath = "/raid/work/yamato/edisk_data/analysis_data/"
uvtabfilename = datafilepath + "L1489IRS_continuum.uvtab"
visibility = np.loadtxt(uvtabfilename, unpack=True)
u, v, real, imag, weight = np.ascontiguousarray(visibility)
print("Done.")


# datafilepath = "/raid/work/yamato/edisk_data/edisk_calibrated_data/"
# uvtabfilename = datafilepath + "L1489IRS_continuum.bin_30s.ms.uvtab"
# visibility = np.loadtxt(uvtabfilename, unpack=True)
# u, v, real, imag, weight, freqs = np.ascontiguousarray(visibility)
# print("Done.")

# datafilepath = "/raid/work/yamato/edisk_data/edisk_calibrated_data/"
# msfilename = datafilepath + "L1489IRS_continuum.bin_30s.ms"
# u, v, real, imag, weight, freqs = import_ms(msfilename)


# get gridding parameters
nxy, dxy, r, rmin, dr = set_grid(u, v)

# setup the fitting parameters
param_name, fixed_param_name, bound, initial_state = setup_params(param_dict)

# function for visibility sampling
def sample_vis(param_dict):

    # retrieve geometrical params
    PA = param_dict.pop("PA", gp_default["PA"]["p0"])
    incl = param_dict.pop("incl", gp_default["incl"]["p0"])
    dRA = param_dict.pop("dRA", gp_default["dRA"]["p0"])
    dDec = param_dict.pop("dDec", gp_default["dDec"]["p0"])

    # get model array
    model = model_func_1d(r / arcsec, **param_dict)

    # sampling by GALARIO
    V = sampleProfile(
        intensity=model,
        Rmin=rmin,
        dR=dr,
        nxy=nxy,
        dxy=dxy,
        u=u,
        v=v,
        dRA=dRA * arcsec,
        dDec=dDec * arcsec,
        PA=PA * deg,
        inc=incl * deg,
        check=False,
    )

    return V


# likelihood function
def log_likelihood(param):

    param_dict = {name: p for name, p in zip(param_name, param)}

    # update fixed param
    param_dict.update({name: param_dict[name]["p0"] for name in fixed_param_name})

    # sample model visibility
    model_vis = sample_vis(param_dict)

    # compute log likelihood
    rms = np.sqrt(1.0 / weight)
    ll = -0.5 * np.sum(
        ((model_vis.real - real) ** 2 + (model_vis.imag - imag) ** 2) / rms**2
        + np.log(2 * pi * rms**2)
    )

    return ll


# probability function
def log_probability(param):
    lp = log_prior(param, bound)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(param)
    return lp + ll


###############################################################


################### Actual MCMC run ###########################
# nwalker = 32
# nstep = 5000
# progress = True


# with multiprocessing.Pool(processes=8) as pool:
#     # pool = None
#     sampler = run_emcee(
#         log_probability=log_probability,
#         initial_state=initial_state,
#         nwalker=nwalker,
#         nstep=nstep,
#         pool=pool,
#         progress=progress,
#         # nthreads=8
#     )

# # save the EmsambleSampler object into a pickle
# with open(datafilepath + "L1489IRS_continuum_sampler_.pkl", "wb") as f:
#     pickle.dump(sampler, f, protocol=pickle.HIGHEST_PROTOCOL)

################################################################


########## plot various figures #############

### TODO: implement the autocorrelation analysis ###
# autocorrelation analysis
# tau = sampler.get_autocorr_time()
# print("Autocorreation time: {:g}".format(tau))

# nburnin = int(3 * tau)
# thin = int(tau / 2)

# load back the EnsambleSampler
import pickle

with open(
    "L1489IRS_continuum_sampler_threeGaussian_5000step.pkl", "rb"
) as f:
    sampler = pickle.load(f)

# get the flatted and discarded sample
nburnin = 2000
thin = 1

sample = sampler.get_chain()
sample_flat = sampler.get_chain(flat=True)
sample_flat_disc = sampler.get_chain(discard=nburnin, thin=thin, flat=True)


# corner plot
corner_fig = corner.corner(
    sample_flat_disc,
    labels=param_name,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
)

# chain plot
walker_figs = plot_walker(sample=sample, nburnin=nburnin, labels=param_name)

# overplot MAP model
MAP_param = sample_flat[np.argmax(sampler.get_log_prob(flat=True))]
param_dict = {name: p for name, p in zip(param_name, MAP_param)}

# update fixed param
param_dict.update({name: param_dict[name]["p0"] for name in fixed_param_name})

# get the visibility for MAP model
MAP_vis = sample_vis(param_dict.copy())

np.save("./L1489IRS_continuum_PointSource_GaussianRing_Gaussian_MAP_vis.npy", MAP_vis)

# real/imag vs. baseline plot for observe values
model_fig = plot_uv(
    u=u,
    v=v,
    real=real,
    imag=imag,
    weight=weight,
    incl=param_dict["incl"],
    PA=param_dict["PA"],
    fmt="o",
    capsize=3,
    markersize=5,
    zorder=-100,
    binsize=10e3,
    uvrange=(0, 5000e3),
)

# deprojection and binning for model visibility
_, _, uvdist_deproj = deproject_vis(u, v, incl=param_dict["incl"], PA=param_dict["PA"])
uvdist_deproj_binned, real_binned, imag_binned = bin_vis(
    uvdist=uvdist_deproj,
    real=MAP_vis.real,
    imag=MAP_vis.imag,
    bins=np.arange(0, 5000e3, 5e3),
    weighted_average=False,
)

# plot model visibility
for ax, V in zip(model_fig.axes, [real_binned, imag_binned]):
    ax.plot(uvdist_deproj_binned / 1e3, V)
    ax.grid()

# ancillary stuffs
model_fig.axes[0].set(ylim=(0.0, 0.05), xlim=(0, 5000))
model_fig.axes[1].set(ylim=(-0.0055, 0.0055), xlim=(0, 5000))

# plt.show()


################## save the figures into a directory #####################
import subprocess


figpath = "./fig_{:s}_{:s}".format(source, model_name)
subprocess.run(["mkdir", figpath])
corner_fig.savefig(figpath + "corner.png", bbox_inches="tight", dpi=300)
for i, fig in enumerate(walker_figs):
    fig.savefig(
        figpath + "walker_{}.png".format(param_name[i]),
        bbox_inches="tight",
        dpi=300,
    )
model_fig.savefig(figpath + "uvplot.png", bbox_inches="tight", dpi=300)
