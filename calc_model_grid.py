from imageanalysis_utils import Gaussian1d, GaussianRing1d
from uvanalysis_utils import set_grid


################# SET UP THE MODEL YOU WANT TO FIT HERE ####################
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