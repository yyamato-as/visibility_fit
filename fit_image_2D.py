import pandas as pd
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.visualization import AsinhStretch, ImageNormalize
import pickle
from analysis_utils import imfit_wrapper


# image path etc.
prefix = "/raid/work/yamato/edisk_data/v0_image/L1489IRS/DDT_LP/continuum/L1489IRS_SBLB_continuum_"
#imaging_param = "robust_1.0_taper_3000klambda"
imaging_param = "robust_1.0"
ext = ".image.tt0.fits"

imagepath = prefix + imaging_param + ext

# image cutout for plot
cutout = (slice(2100, 3900), slice(2100,3900))

# plot parameters
norm = ImageNormalize(fits.getdata(imagepath)[cutout], vmin=0.0, stretch=AsinhStretch(a=0.03))
imshow_kw = pcolorfast_kw = {'cmap': 'inferno', 'norm': norm}
contour_kw = {"colors": "white", "linewidths": 0.2, "linestyles": "dashed"}

# nominal source position
common_dir = "04h04m43.070001s +26d18m56.20011s" # from imaging script

# mask setup 
mask_ra = common_dir.split()[0].replace("h", ":").replace("m", ":").replace("s", "")
mask_dec = common_dir.split()[1].replace("d", ".").replace("m", ".").replace("s", "")
mask_pa = 0.0  # position angle of mask in degrees
mask_maj = 5.0  # semimajor axis of mask in arcsec
mask_min = 5.0  # semiminor axis of mask in arcsec
common_mask = "ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]" % (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)

# set the initial estimates file
# peak intensity, peak xpixel, peak ypixel, maj, min, pa
# values from Sai et al. 2020
est_str_list = [
    "0.003, 3000, 3000, 0.097arcsec, 0.037arcsec, 49deg\n",
    "0.001, 3000, 3000, 4.1arcsec, 1.2arcsec, 69deg\n",
]  # need \n

analysis_data_path = "/raid/work/yamato/edisk_data/analysis_data/"
estimates_filename = analysis_data_path + "L1489IRS_cont_imfit.estimates"
with open(estimates_filename, "w") as f:
    f.writelines(est_str_list)

# set model and residual file
mod_filename = analysis_data_path + "L1489IRS_cont_imfit.model.image"
res_filename = analysis_data_path + "L1489IRS_cont_imfit.residual.image"

# import the statistics
stat_file = analysis_data_path + "L1489IRS_continuum_rms.pkl"
stat = pd.read_pickle(stat_file)
rms = stat.loc[imaging_param, "rms [mJy/beam]"] * 1e-3 # in Jy / beam


# 2 component Gaussian fit
result = imfit_wrapper(
    imagepath,
    region=common_mask,
    model=mod_filename,
    residual=res_filename,
    estimates=estimates_filename,
    rms=rms,
    plot=True,
    plot_region_slices=cutout,
    plot_kw={'imshow_kw': imshow_kw},
)

plt.savefig("./fig_image_analysis/L1489IRS_continuum_2D_GaussianFit_2comp.png", bbox_inches='tight')

plt.show()

# save the result
savefile = analysis_data_path + "L1489IRS_cont_imfit.result.pkl"
with open(savefile, 'wb') as f:
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
