import numpy as np
import casatools
import astropy.constants as ac
import astropy.units as u
import sys
sys.path.append("/home/yamato/Project/visibility_fit")
from uvanalysis_utils import plot_uv
import matplotlib.pyplot as plt

c = ac.c.to(u.m/u.s).value

ms = casatools.ms()


# msfilename = "/raid/work/yamato/edisk_data/edisk_calibrated_data_old/archive_LP/L1489IRS_SB1_continuum.ms.split_1chanPerSpw"
msfilename = "/raid/work/yamato/edisk_data/edisk_calibrated_data_old/archive_LP/L1489IRS_SB1_continuum.ms.split_keepflagsFalse_originalNchans"
# msfilename = "/raid/work/yamato/edisk_data/edisk_calibrated_data_old/L1489IRS_SB1_continuum.ms.split"

ms.open(msfilename)

print("Loading {:s}...".format(msfilename))

spw = [key for key in ms.getspectralwindowinfo()]

data = {}
for i in spw:
    ms.selectinit(datadescid=int(i))
    #ms.selectpolarization(corr)
    data[i] = ms.getdata(["u" ,"v", "data", "weight", "axis_info"])
    ms.reset()

ms.close()

# manipulate read visibilities
u = []
v = []
V = []
weight = []
freqs = []

for spw in data.keys():
    print(spw, data[spw]["data"].shape)
    
    # average over polarization
    if data[spw]["data"].shape[0] == 2:
        _V = np.sum(data[spw]["data"]*data[spw]["weight"][:,None,:], axis=0) / np.sum(data[spw]["weight"], axis=0)
        _weight = np.sum(data[spw]["weight"], axis=0)
        # V_XX = data[spw]["data"][0,:,:]
        # V_YY = data[spw]["data"][1,:,:]
        # weight_XX = data[spw]["weight"][0,:]
        # weight_YY = data[spw]["weight"][1,:]
        # _weight = weight_XX + weight_YY
        # _V = (V_XX * weight_XX + V_YY * weight_YY) / _weight

    else:
        _weight = data[spw]["weight"][0,:]
        _V = data[spw]["data"][0,:]

    nchan, nuv = _V.shape
    _freqs = data[spw]["axis_info"]["freq_axis"]["chan_freq"]

    _freqs = np.tile(_freqs, nuv)
    _wles = c / _freqs
    _u = np.tile(data[spw]["u"], (nchan, 1)) / _wles # in lmabda
    _v = np.tile(data[spw]["v"], (nchan, 1)) / _wles # in lambda
    _weight = np.tile(_weight, (nchan, 1))

    # remove the autocorrelation; here all the variables are flattened
    # -> but this removement causes an issue if you want the model visibility to get back the original ms file after fitting... so quit
    # cc = uvdist != 0

    # u = u[cc]
    # v = v[cc]
    # V = V[cc]
    # weight = weight[cc]
    # freqs = freqs[cc]

    # append each component with flatten
    u.append(_u.ravel())
    v.append(_v.ravel())
    V.append(_V.ravel())
    weight.append(_weight.ravel())
    freqs.append(_freqs.ravel())

# concatenate 
u = np.ascontiguousarray(np.concatenate(u))
v = np.ascontiguousarray(np.concatenate(v))
V = np.concatenate(V)
real = np.ascontiguousarray(V.real)
imag = np.ascontiguousarray(V.imag)
weight = np.ascontiguousarray(np.concatenate(weight))
freqs = np.ascontiguousarray(np.concatenate(freqs))

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

