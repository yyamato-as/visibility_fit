import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.visualization import AsinhStretch, ImageNormalize
from imageanalysis_utils import (
    plot_2D_map,
    get_projected_coord,
    get_radec_coord,
    calc_radial_profile,
    get_beam_solid_angle
)
import pickle

prefix = "/raid/work/yamato/edisk_data/v0_image/L1489IRS/DDT_LP/continuum/L1489IRS_SBLB_continuum_"
imaging_param = "robust_1.0"
ext = ".image.tt0.fits"

imagepath = prefix + imaging_param + ext

# image cutout for plot
cutout = (slice(2100, 3900), slice(2100, 3900))

# plot parameters
norm = ImageNormalize(
    fits.getdata(imagepath)[cutout], vmin=0.0, stretch=AsinhStretch(a=0.03)
)
imshow_kw = pcolorfast_kw = {"cmap": "inferno", "norm": norm}
contour_kw = {"colors": "white", "linewidths": 0.2, "linestyles": "dashed"}

# load the imfit result
savefile = "/raid/work/yamato/edisk_data/analysis_data/L1489IRS_cont_imfit.result.pkl"
with open(savefile, "rb") as f:
    result = pickle.load(f)

header = fits.getheader(imagepath)
data = fits.getdata(imagepath)

beam = (header["BMAJ"] * 3600, header["BMIN"] * 3600, 90 - header["BPA"])  # in arcsec
scale = (50.0 / 140.0, "50 au")  # in arcsec

# Omega_beam = get_beam_solid_angle(beam[:-1])
# data /= Omega_beam # in Jy /sr

# source center coordinate from Gaussian fit
center_coord = result["component0"]["peak"]
# center_coord = SkyCoord(common_dir, frame='icrs')

# calculate the coordinate
x, y = get_radec_coord(header, center_coord=center_coord)
r, theta = get_projected_coord(
    header, PA=68, incl=76, center_coord=center_coord, which="polar"
)

# plot to check the coordinate
fig, ax = plt.subplots()
plot_2D_map(
    data[cutout],
    X=x[cutout[0]],
    Y=y[cutout[1]],
    ax=ax,
    contour=False,
    beam=beam,
    scale=scale,
    pcolorfast_kw={"norm": norm, "cmap": "inferno"},
)
ax.contour(
    x[cutout[0]], y[cutout[1]], r[cutout], levels=20, colors="white", linewidths=0.5
)
ax.contour(
    x[cutout[0]],
    y[cutout[1]],
    theta[cutout],
    levels=20,
    colors="white",
    linestyles="solid",
    linewidths=0.5,
)
ax.set(xlabel=r"$\Delta\alpha\cos(\delta)$ [arcsec]", ylabel="$\Delta\delta$ [arcsec]")

# plot the radial profile
PA = result["component1"]["size"]["pa"].value
incl = result["component1"]["inclination"].value
rmax = 5.0  # outer boundary of calculation region
rbins = np.arange(
    header["BMAJ"] * 3600 * 0.25 * 0.5, rmax, header["BMAJ"] * 3600 * 0.25
)
include_theta = 90.0

_, radprof, radprof_scatter = calc_radial_profile(
    header,
    data,
    PA=PA,
    incl=incl,
    center_coord=center_coord,
    rbins=rbins,
    include_theta=include_theta,
)

fig, ax = plt.subplots()
ax.plot(rbins, radprof, label="wedge = {:.0f} deg".format(include_theta))
ax.fill_between(rbins, radprof - radprof_scatter, radprof + radprof_scatter, alpha=0.3)
ax.axhline(y=0.0, color="grey", ls="dashed")
ax.set(
    xlim=(0, rmax),
    xlabel="Radius [arcsec]",
    ylabel=r"$I_{\nu}$ [Jy / beam]",
    yscale="log",
    ylim=(5e-6, 5e-3),
)
ax.legend()


########### FITIING THE RADIAL PROFILE BY ANALYTICAL FUNCTION ##########
from imageanalysis_utils import Gaussian1d, GaussianRing1d
from scipy.optimize import curve_fit

def model_1d(r, F_c, sigma_c, r0_r, F_r, sigma_r, F_b, sigma_b):
    return Gaussian1d(r, F_c, sigma_c) + GaussianRing1d(r, r0_r, F_r, sigma_r) + Gaussian1d(r, F_b, sigma_b)

# fit
popt, pcov = curve_fit(model_1d, rbins, radprof, p0=(-3., 0.038, 0.43, -4., 0.2, -3., 1.5))

ax.plot(rbins, model_1d(rbins, *popt))
ax.plot(rbins, Gaussian1d(rbins, *popt[:2]), ls='dashed', color='grey')
ax.plot(rbins, GaussianRing1d(rbins, *popt[2:5]), ls='dashed', color='grey')
ax.plot(rbins, Gaussian1d(rbins, *popt[5:]), ls="dashed", color='grey')

fig.savefig("./fig_image_analysis/L1489IRS_continuum_radial_profile_fit.png", bbox_inches='tight', dpi=300)
plt.show()






