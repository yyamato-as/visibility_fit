import glob
import numpy as np
import pandas as pd
import casatasks
import casatools
import casadata
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.visualization import AsinhStretch, ImageNormalize
from analysis_utils import fetch_beam_info, jypb_to_K_astropy, jypb_to_K, jypb_to_K_RJ
import warnings

#%matplotlib widget

pd.options.display.float_format = "{:.3g}".format


# fetch the image paths
prefix = (
    "/raid/work/yamato/edisk_data/v0_image/L1489IRS/DDT_LP/continuum/L1489IRS_SBLB_continuum_"
)
imaging_param = [
    "robust_-2.0",
    "robust_-1.0",
    "robust_-0.5",
    "robust_0.0",
    "robust_0.5",
    "robust_1.0",
    "robust_2.0",
    "robust_1.0_taper_1000klambda",
    "robust_1.0_taper_2000klambda",
    "robust_1.0_taper_3000klambda",
    "robust_2.0_taper_1000klambda",
    "robust_2.0_taper_2000klambda",
    "robust_2.0_taper_3000klambda",
]
ext = ".pbcor.tt0.fits"

# parameters for noise mask
common_dir = "04h04m43.070001s +26d18m56.20011s"  # from imaging script
mask_ra = common_dir.split()[0].replace("h", ":").replace("m", ":").replace("s", "")
mask_dec = common_dir.split()[1].replace("d", ".").replace("m", ".").replace("s", "")
r_in = 6.0  # in arcsec
r_out = 8.0  # in arcsec

# noise annulus as CASA region format
noise_annulus = "annulus[[%s, %s],['%.1farcsec', '%.1farcsec']]" % (
    mask_ra,
    mask_dec,
    r_in,
    r_out,
)

# initialize table
table = pd.DataFrame(columns=['rms [mJy/beam]', 'rms [K]'])

# measure the rms over the mask; use RJ approximation
for ip in imaging_param:
    imagepath = prefix + ip + ext
    beam = fetch_beam_info(imagepath)
    rms = casatasks.imstat(imagepath, region=noise_annulus)["rms"][0] # in Jy/beam 
    nu = 220e9 # TODO: better to fetch the freqyuency from fits header: need to use exportfits with dropdeg=False; ask John?
    rms_K = jypb_to_K_RJ(rms, nu, beam[:-1])
    table.loc[ip] = [rms*1e3, rms_K]

print(table)

# save into a pickle
savefile = "/raid/work/yamato/edisk_data/analysis_data/L1489IRS_continuum_rms.pkl"
table.to_pickle(savefile)
