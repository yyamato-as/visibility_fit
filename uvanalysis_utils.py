### visibility manipulation/analysis relevant ###

import numpy as np
from galario.double import get_image_size
import matplotlib.pyplot as plt
from scipy import stats


def set_grid(u, v, verbose=True):
    # gridding parameter
    nxy, dxy = get_image_size(u, v, verbose=verbose)  # in rad

    # condition for GALARIO interpolation: dxy/2 - Rmin > 5 * dR
    # not to make dR too small, adopt dxy/2/1000 as Rmin
    rmin = dxy / 2.0 * 1e-3  # in rad

    # dR = (dxy/2 - Rmin) / 5.1
    dr = (dxy / 2.0 - rmin) / 5.1  # in rad

    r_pad = 2 * dxy  # padding parameter
    rmax = dxy * nxy / np.sqrt(2) + r_pad  # in rad

    # radial grid on image plane
    r = np.arange(rmin, rmax, dr)  # in rad

    return nxy, dxy, r, rmin, dr


def plot_uv(
    u,
    v,
    real,
    imag,
    weight,
    incl=0.0,
    PA=0.0,
    uvrange=(0, 5000e3),
    binsize=10e3,
    logbin=False,
    fig=None,
    **errorbar_kwargs
):

    if fig is None:
        fig, axes = plt.subplots(
            nrows=2, ncols=1, gridspec_kw={"height_ratios": [4, 1]}, sharex=True
        )

    else:
        axes = fig.axes

    # apply the deprojection
    _, _, uvdist_deproj = deproject_vis(u, v, incl=incl, PA=PA)

    # binning the uv data
    if uvrange is None:
        uvrange = (np.nanmin(uvdist_deproj), np.nanmax(uvdist_deproj))
    if binsize is None:
        binsize = uvrange[1] / 100

    bins = np.arange(uvrange[0], uvrange[1], binsize)  # in lambda
    if logbin:
        bins = 10**bins

    (
        uvdist_deproj_binned,
        real_binned,
        real_binned_err,
        imag_binned,
        imag_binned_err,
    ) = bin_vis(uvdist_deproj, real, imag, weight, bins=bins)

    # plot
    axes[0].errorbar(
        uvdist_deproj_binned / 1e3,  # in kilolambda
        real_binned,
        yerr=real_binned_err,
        **errorbar_kwargs
    )
    axes[1].errorbar(
        uvdist_deproj_binned / 1e3,  # in kilolambda
        imag_binned,
        yerr=imag_binned_err,
        **errorbar_kwargs
    )

    # label etc.
    axes[0].set(ylabel="Real [Jy]")
    axes[1].set(xlabel=r"Baseline [k$\lambda$]", ylabel="Imaginary [Jy]")

    return fig


def bin_vis(
    uvdist,
    real,
    imag,
    weight=None,
    bins=None,
    weighted_average=True,
):

    uvdist_deproj_binned, _, _ = stats.binned_statistic(
        uvdist, uvdist, bins=bins, statistic="mean"
    )

    if weighted_average:
        assert weight is not None
        real_binned, real_binned_err, _, _ = bin_weighted_average(
            uvdist, real, weight, bins=bins
        )
        imag_binned, imag_binned_err, _, _ = bin_weighted_average(
            uvdist, imag, weight, bins=bins
        )
        return (
            uvdist_deproj_binned,
            real_binned,
            real_binned_err,
            imag_binned,
            imag_binned_err,
        )

    else:
        real_binned, _, _ = stats.binned_statistic(
            uvdist, real, bins=bins, statistic="mean"
        )
        imag_binned, _, _ = stats.binned_statistic(
            uvdist, imag, bins=bins, statistic="mean"
        )
        return uvdist_deproj_binned, real_binned, imag_binned


def deproject_vis(u, v, incl=0.0, PA=0.0):
    incl = np.radians(incl)
    PA = np.radians(PA)

    u_deproj = (u * np.cos(PA) - v * np.sin(PA)) * np.cos(incl)
    v_deproj = u * np.sin(PA) + v * np.cos(PA)

    uvdist_deproj = np.hypot(u_deproj, v_deproj)

    return u_deproj, v_deproj, uvdist_deproj


def bin_weighted_average(x, y, weights, bins, std_err=False):
    w, edge_w, num_w = stats.binned_statistic(x, weights, bins=bins, statistic="sum")
    yw, edge_yw, num_yw = stats.binned_statistic(
        x, y * weights, bins=bins, statistic="sum"
    )

    assert np.all(edge_w == edge_yw)
    assert np.all(num_w == num_yw)

    if std_err:
        err, edge_e, num_e = stats.binned_statistic(x, y, bins=bins, statistic="std")
        assert np.all(edge_e == edge_yw)
        assert np.all(num_e == num_yw)
    else:
        err = 1.0 / np.sqrt(w)

    return yw / w, err, edge_yw, num_yw
