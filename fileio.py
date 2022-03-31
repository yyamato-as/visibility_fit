import casatools 
import numpy as np
import shutil
import astropy.constants as ac
import astropy.units as u

c = ac.c.to(u.m/u.s).value

tb = casatools.table()
ms = casatools.ms()

def import_ms(msfilename, export_uvtable=True, filename=None):
    """Import the visibility from a CASA measurement set as 1D arrays using casatools. 

    Parameters
    ----------
    msfilename : str
        The measurement set filename which you want to import.
    export_uvtable : bool, optional
        If write the output into a text file (similar to UVTable), by default True
    filename : str or None, optional
        The filename of output text file, by default None. Relevant only when export_uvtable=True.

    Returns
    -------
    u, v, real, imag, weight, freqs: six 1D numpy array
        visibility
    """

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
            V_XX = data[spw]["data"][0,:,:]
            V_YY = data[spw]["data"][1,:,:]
            weight_XX = data[spw]["weight"][0,:]
            weight_YY = data[spw]["weight"][1,:]
            _weight = weight_XX + weight_YY
            _V = (V_XX * weight_XX + V_YY * weight_YY) / _weight

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

    print("Done.")

    if export_uvtable:
        if filename is None:
            filename = "output.uvtab"

        np.savetxt(
            filename,
            np.column_stack([u, v, real, imag, weight, freqs]),
            fmt="%10.6e",
            delimiter="\t",
            header="u [lambda]\t v [lambda]\t real [Jy] \t imag [Jy]\t weight \t nu [Hz]",
        )

    return u, v, real, imag, weight, freqs


def export_ms(basems, outms, real, imag, weight):
    """Export the visibility (1D arrays) to a CASA measurement set file

    Parameters
    ----------
    basems : str
        Measurement set filename from which you get the data by import_ms function. Must have the same number of data points as input.
    outms : str
        The filename to which you want to export.
    real : 1D numpy array
        Real part of the visibility.
    imag : 1D numpy array
        Imaginary part of the visibility.
    weight : 1D numpy array
        Weight.
    """

    shutil.copytree(basems, outms)

    assert real.shape == imag.shape == weight.shape
    assert real.ndim == imag.ndim == weight.ndim == 1

    datasize = len(real)

    ms.open(outms, nomodify=False)

    spw = [key for key in ms.getspectralwindowinfo()]

    # check the number of datapoint consistency
    ndata = {}
    nchan = {}
    spw_array = []
    for i in spw:
        ms.selectinit(datadescid=int(i))
        #ms.selectpolarization(corr)

        rec = ms.getdata(["data"])

        nc = rec["data"].shape[1]
        nd = rec["data"].shape[2]

        nchan[i] = nc
        ndata[i] = nc * nd

        spw_array.append(np.full(ndata[i], int(i)))

        ms.reset()

    spw_array = np.concatenate(spw_array)

    if datasize != np.sum([i for i in ndata.values()]):
        raise ValueError("Data size is not consistent with the base measurement set.")

    # put the data onto each spectral window
    print("Exporting into {:s}...".format(outms))

    V = real + imag*1.0j

    for i in spw:
        print("processing spw="+i)
        ms.selectinit(datadescid=int(i))
        #ms.selectpolarization(corr)

        rec = ms.getdata(["data", "weight"])

        # print(rec["data"].shape, rec["weight"].shape)

        v = V[spw_array == int(i)].reshape(nchan[i], -1)
        w = weight[spw_array == int(i)].reshape(nchan[i], -1)

        if rec["data"].shape[0] == 2 and rec["data"].ndim == 3:
            #rec["data"] = np.tile(v, (2, 1, 1))
            rec["data"][0,:,:] = v 
            rec["data"][1,:,:] = v
            #rec["weight"] = np.tile(np.mean(w, axis=0), (2, 1, 1))
            rec["weight"][0,:] = np.mean(w, axis=0) 
            rec["weight"][1,:] = np.mean(w, axis=0)

        else:
            rec["data"][0,:,:] = v
            rec["weight"][0,:] = np.mean(w, axis=0)

        # print(rec["data"].shape, rec["weight"].shape)

        ms.putdata(rec)
        ms.reset()

    ms.close()

    print("Writing done.")



if __name__ == '__main__':
    import numpy as np
    # from fileio import export_ms

    datafilepath = "/raid/work/yamato/edisk_data/edisk_calibrated_data/"
    msfilename = datafilepath + "L1489IRS_SB1_continuum.bin_30s.ms"
    u, v, real, imag, weight, freqs = import_ms(msfilename, export_uvtable=True, filename=msfilename+".uvtab")

    # MAP_vis = np.load("./L1489IRS_SB1_continuum_PointSource_GaussianRing_Gaussian_MAP_vis.npy")

    export_ms(
        basems="/raid/work/yamato/edisk_data/edisk_calibrated_data/L1489IRS_SB1_continuum.bin_30s.ms",
        outms="/raid/work/yamato/edisk_data/edisk_calibrated_data/L1489IRS_SB1_continuum.bin_30s.model.ms",
        real=real,
        imag=imag,
        weight=np.ones(real.shape),
)

    