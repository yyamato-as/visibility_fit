from fileio import import_mms
from uvanalysis_utils import plot_uv
import matplotlib.pyplot as plt

msfilepath = "/raid/work/yamato/edisk_data/edisk_calibrated_data_old/archive_LP/"
msfilepath = "/raid/work/yamato/edisk_data//L1489IRS/eDisk_calibrated_data/"

# u = []
# v = []
# real = []
# imag = []
# weight = []
# for i in ["SB1", "SB2", "LB1", "LB2"]:
# i = "SB1"
# msfilename = msfilepath + "L1489IRS_{}_continuum.ms".format(i)
vislist = [msfilepath + "L1489IRS_{}_continuum.ms".format(i) for i in ["LB2"]]

u, v, real, imag, weight, freqs = import_mms(vislist=vislist, export_uvtable=False)

print(u.shape)

    # u.append(_u)
    # v.append(_v)
    # real.append(_real)
    # imag.append(_imag)


fig = plot_uv(
    u,
    v,
    real,
    imag,
    weight,
    incl=73.0,
    PA=69.0,
    fmt="o",
    capsize=3,
    markersize=5,
    zorder=-100,
    binsize=10e3,
    uvrange=(0, 5000e3)
)

plt.show()
