import numpy as np
from fileio import export_ms

MAP_vis = np.load("./L1489IRS_SB1_continuum_PointSource_GaussianRing_Gaussian_MAP_vis.npy")

export_ms(
    basems="/raid/work/yamato/edisk_data/edisk_calibrated_data/L1489IRS_SB1_continuum.bin_30s.ms",
    outms="/raid/work/yamato/edisk_data/edisk_calibrated_data/L1489IRS_SB1_continuum.bin_30s.model.ms",
    real=MAP_vis.real,
    imag=MAP_vis.imag,
    weight=np.ones(MAP_vis.shape),
)


# tclean again
