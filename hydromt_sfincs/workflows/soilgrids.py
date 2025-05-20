import logging

import numpy as np
import xarray as xr
import pandas as pd

from typing import List

logger = logging.getLogger(__name__)

def concat_layers(
    ds: xr.Dataset,
    soil_fn: str = "soilgrids",
    variables: List[str] = ["bd", "oc", "ph", "clyppt", "sltppt", "sndppt"],
):
    """
    Preprocess functions to concat soilgrids along a layer dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing soil properties.
    soil_fn : str
        soilgrids version {'soilgrids', 'soilgrids_2020'}
    variables : list
        List of soil properties to concat.
    """
    if soil_fn == "soilgrids_2020":
        nb_sl = 6
    else:
        nb_sl = 7
    ds = ds.assign_coords(sl=np.arange(1, nb_sl + 1))

    for var in variables:
        da_prop = []
        for i in np.arange(1, nb_sl + 1):
            da_prop.append(ds[f"{var}_sl{i}"])
            # remove layer from ds
            ds = ds.drop_vars(f"{var}_sl{i}")
        da = xr.concat(
            da_prop,
            pd.Index(np.arange(1, nb_sl + 1, dtype=int), name="sl"),
        ).transpose("sl", ...)
        da.name = var
        # add concat maps to ds
        ds[f"{var}"] = da

    return ds

def kv_brakensiek(thetas, clay, sand):
    """
    Determine saturated hydraulic conductivity kv [mm/day].

    Based on:
      Brakensiek, D.L., Rawls, W.J.,and Stephenson, G.R.: Modifying scs hydrologic
      soil groups and curve numbers for range land soils, ASAE Paper no. PNR-84-203,
      St. Joseph, Michigan, USA, 1984.

    Parameters
    ----------
    thetas: float
        saturated water content [m3/m3].
    clay : float
        clay percentage [%].
    sand: float
        sand percentage [%].

    Returns
    -------
    kv : float
        saturated hydraulic conductivity [mm/day].

    """
    kv = (
        np.exp(
            19.52348 * thetas
            - 8.96847
            - 0.028212 * clay
            + (1.8107 * 10**-4) * sand**2
            - (9.4125 * 10**-3) * clay**2
            - 8.395215 * thetas**2
            + 0.077718 * sand * thetas
            - 0.00298 * sand**2 * thetas**2
            - 0.019492 * clay**2 * thetas**2
            + (1.73 * 10**-5) * sand**2 * clay
            + 0.02733 * clay**2 * thetas
            + 0.001434 * sand**2 * thetas
            - (3.5 * 10**-6) * clay**2 * sand
        )
        * (2.78 * 10**-6)
        * 1000
        * 3600
        * 24
    )

    return kv

def kv_cosby(sand, clay):
    """
    Determine saturated hydraulic conductivity kv [mm/day].

    based on:
      Cosby, B.J., Hornberger, G.M., Clapp, R.B., Ginn, T.R., 1984.
      A statistical exploration of the relationship of soil moisture characteristics to
      the physical properties of soils. Water Resour. Res. 20(6) 682-690.

    Parameters
    ----------
    sand: float
        sand percentage [%].
    clay : float
        clay percentage [%].

    Returns
    -------
    kv : float
        saturated hydraulic conductivity [mm/day].

    """
    kv = 60.96 * 10.0 ** (-0.6 + 0.0126 * sand - 0.0064 * clay) * 10.0

    return kv

def thetas_toth(ph, bd, clay, silt):
    """
    Determine saturated water content [m3/m3].

    Based on:
      Tóth, B., Weynants, M., Nemes, A., Makó, A., Bilas, G., and Tóth, G.:
      New generation of hydraulic pedotransfer functions for Europe, Eur. J.
      Soil Sci., 66, 226–238. doi: 10.1111/ejss.121921211, 2015.

    Parameters
    ----------
    ph: float
        pH [-].
    bd : float
        bulk density [g /cm3].
    clay: float
        clay percentage [%].
    silt: float
        silt percentage [%].

    Returns
    -------
    thetas : float
        saturated water content [cm3/cm3].

    """
    thetas = (
        0.5653
        - 0.07918 * bd**2
        + 0.001671 * ph**2
        + 0.0005438 * clay
        + 0.001065 * silt
        + 0.06836
        - 0.00001382 * clay * ph**2
        - 0.00001270 * silt * clay
        - 0.0004784 * bd**2 * ph**2
        - 0.0002836 * silt * bd**2
        + 0.0004158 * clay * bd**2
        - 0.01686 * bd**2
        - 0.0003541 * silt
        - 0.0003152 * ph**2
    )

    return thetas

def average_soillayers(ds, soilthickness):
    """
    Determine weighted average of soil property at different depths over soil thickness.

    Using the trapezoidal rule.
    See also: Hengl, T., Mendes de Jesus, J., Heuvelink, G. B. M., Ruiperez Gonzalez,
    M., Kilibarda, M., Blagotic, A., et al.: SoilGrids250m: \
Global gridded soil information based on machine learning,
    PLoS ONE, 12, https://doi.org/10.1371/journal.pone.0169748, 2017.
    This function is used for soilgrids (2017).

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil property at each soil depth [sl1 - sl7].
    soilthickness : xarray.DataArray
        Dataset containing soil thickness [cm].

    Returns
    -------
    da_av : xarray.DataArray
        Dataset containing the weighted average of the soil property.

    """
    da_sum = soilthickness * 0.0
    # set NaN values to 0.0 (to avoid RuntimeWarning in comparison soildepth)
    d = soilthickness.fillna(0.0)

    soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])

    for i in range(1, len(ds.sl)):  # range(1, 7):
        da_sum = da_sum + (
            (soildepth_cm[i] - soildepth_cm[i - 1])
            * (ds.sel(sl=i) + ds.sel(sl=i + 1))
            * (d >= soildepth_cm[i])
            + (d - soildepth_cm[i - 1])
            * (ds.sel(sl=i) + ds.sel(sl=i + 1))
            * ((d < soildepth_cm[i]) & (d > soildepth_cm[i - 1]))
        )

    da_av = xr.apply_ufunc(
        lambda x, y: x * (1 / (y * 2)),
        da_sum,
        soilthickness,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    da_av.raster.set_nodata(np.nan)

    return da_av

def average_soillayers_block(ds, soilthickness):
    """
    Determine weighted average of soil property at different depths over soil thickness.

    Assuming that the properties are computed at the mid-point of the interval and
    are considered constant over the whole depth interval (Sousa et al., 2020).
    https://doi.org/10.5194/soil-2020-65
    This function is used for soilgrids_2020.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil property over each soil depth profile [sl1 - sl6].
    soilthickness : xarray.DataArray
        Dataset containing soil thickness [cm].

    Returns
    -------
    da_av : xarray.DataArray
        Dataset containing weighted average of soil property.

    """
    da_sum = soilthickness * 0.0
    # set NaN values to 0.0 (to avoid RuntimeWarning in comparison soildepth)
    d = soilthickness.fillna(0.0)

    soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])

    for i in ds.sl:
        da_sum = da_sum + (
            (soildepth_cm[i] - soildepth_cm[i - 1])
            * ds.sel(sl=i)
            * (d >= soildepth_cm[i])
            + (d - soildepth_cm[i - 1])
            * ds.sel(sl=i)
            * ((d < soildepth_cm[i]) & (d > soildepth_cm[i - 1]))
        )

    da_av = xr.apply_ufunc(
        lambda x, y: x / y,
        da_sum,
        soilthickness,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    da_av.raster.set_nodata(np.nan)

    return da_av

def thetar_rawls_brakensiek(sand, clay, thetas):
    """
    Determine residual water content [m3/m3].

    Based on:
      Rawls,W. J., and Brakensiek, D. L.: Estimation of SoilWater Retention and
      Hydraulic Properties, In H. J. Morel-Seytoux (Ed.),
      Unsaturated flow in hydrologic modelling - Theory and practice, NATO ASI Series 9,
      275–300, Dordrecht, The Netherlands: Kluwer Academic Publishing, 1989.

    Parameters
    ----------
    sand : float
        sand percentage [%].
    clay: float
        clay percentage [%].
    thetas: float
        saturated water content [m3/m3].

    Returns
    -------
    thetar : float
        residual water content [m3/m3].

    """
    thetar = (
        -0.0182482
        + 8.7269 * 10**-4 * sand
        + 0.00513488 * clay
        + 0.02939286 * thetas
        - 1.5395 * 10**-4 * clay**2
        - 1.0827 * 10**-3 * sand * thetas
        - 1.8233 * 10**-4 * clay**2 * thetas**2
        + 3.0703 * 10**-4 * clay**2 * thetas
        - 2.3584 * 10**-3 * thetas**2 * clay
    )

    return thetar

def kv_layers(ds, thetas, ptf_name = "brakensiek"):
    """
    Determine vertical saturated hydraulic conductivity (KsatVer) per soil layer depth.

    Based on PTF.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil properties at each soil depth [sl1 - sl7].
    thetas: xarray.Dataset
        Dataset containing thetaS at each soil layer depth.
    ptf_name : str
        PTF to use for calculation KsatVer.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing KsatVer [mm/day] for each soil layer depth.
    """
    if ptf_name == "brakensiek":
        ds_out = xr.apply_ufunc(
            kv_brakensiek,
            thetas,
            ds["clyppt"],
            ds["sndppt"],
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )
    elif ptf_name == "cosby":
        ds_out = xr.apply_ufunc(
            kv_cosby,
            ds["clyppt"],
            ds["sndppt"],
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )

    ds_out.name = "kv"
    ds_out.raster.set_nodata(np.nan)

    return ds_out

def air_entry_pressure(clay, silt):
    """
    Determine air entry pressure

    Parameters
    ----------
    clay: float
        sand percentage [%].
    silt: float
        silt percentage [%].

    Returns
    -------
    air_entry_pressure : float
        air entry pressure [cm]


    """
    sand = 100 - (clay + silt)

    soil_texture = np.where(
        np.logical_and(clay >= 40.0, sand >= 20.0, sand <= 45),
        353,  # clay
        np.where(
            np.logical_and(clay >= 27.0, sand >= 20.0, sand <= 45),
            88.0,  # clay loam
            np.where(
                np.logical_and(silt <= 40.0, sand <= 20.0),
                353,  # clay
                np.where(
                    np.logical_and(silt > 40.0, clay >= 40.0),
                    340,  # silty clay
                    np.where(
                        np.logical_and(clay >= 35.0, sand >= 45.0),
                        50.7,  # sandy clay
                        np.where(
                            np.logical_and(clay >= 27.0, sand < 20.0),
                            132,  # silty clay loam
                            np.where(
                                np.logical_and(clay <= 10.0, silt >= 80.0),
                                68.1,  # silt
                                np.where(
                                    (silt >= 50.0),
                                    70.3,  # silt loam
                                    np.where(
                                        np.logical_and(
                                            clay >= 7.0, sand <= 52.0, silt >= 28.0
                                        ),
                                        38.9,  # loam
                                        np.where(
                                            (clay >= 20.0),
                                            26.2,  # sandy clay loam
                                            np.where(
                                                (clay >= (sand - 70)),
                                                17.7,  # sandy loam
                                                np.where(
                                                    (clay >= (2 * sand - 170.0)),
                                                    9.58,  # loamy sand
                                                    np.where(
                                                        np.isnan(clay), np.nan, 7.02
                                                    ),  # sand
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    return soil_texture