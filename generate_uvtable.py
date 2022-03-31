from fileio import import_ms
from uvanalysis_utils import plot_uv
from casatasks import split, concat
import matplotlib.pyplot as plt
import subprocess

incl, PA = 73, 69


msfilepath = "/raid/work/yamato/edisk_data/edisk_calibrated_data/DDT_LP/"
array_config = ["SB1", "SB2", "LB1", "LB2"]
msfilelist = [
    msfilepath + "L1489IRS_{:s}_continuum.ms".format(ac) for ac in array_config
]


### time averaging over 30s ###
# eliminate flagged data
# time averaging will go across scans and sub-scans

msfilelist_split = []

for msfile in msfilelist:
    outputvis = msfile.replace(".ms", ".bin_30s.ms")
    subprocess.run(["rm", "-r", outputvis])
    print("Splitting out {}...".format(msfile))
    split(
        vis=msfile,
        outputvis=outputvis,
        datacolumn="data",
        keepflags=False,
        timebin="30s",
        #combine="state,scan",
    )
    msfilelist_split.append(outputvis)

### concatenate into on ms ###
concatvis = msfilepath + "L1489IRS_continuum.ms"
subprocess.run(["rm", "-r", concatvis])
print("Concatenating {}...".format(",".join(msfilelist_split)))
concat(vis=msfilelist_split, concatvis=concatvis, dirtol="0.01arcsec")


### export to uvtable ###
uvtablepath = "/raid/work/yamato/edisk_data/analysis_data/"
filename = uvtablepath + concatvis.split("/")[-1] + ".uvtab"
u, v, real, imag, weight, freqs = import_ms(
    concatvis, export_uvtable=True, filename=filename
)


### test plot uv dist vs. amp
fig = plot_uv(
    u=u,
    v=v,
    real=real,
    imag=imag,
    weight=weight,
    incl=incl,
    PA=PA,
    fmt="o",
    capsize=3,
    markersize=5,
    zorder=-100,
    binsize=10e3,
    uvrange=(0, 5000e3),
)

for ax in fig.axes:
    ax.grid()

fig.axes[0].set(ylim=(0.0, 0.05), xlim=(0, 5000))
fig.axes[1].set(ylim=(-0.0055, 0.0055), xlim=(0, 5000))

fig.savefig("./uvplot.pdf", bbox_inches="tight", pad_inches=0.01)

plt.show()
