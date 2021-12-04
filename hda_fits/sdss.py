import urllib
from io import BytesIO
from typing import List, Union

import numpy as np
import pandas as pd
import requests
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.io.fits.hdu.image import PrimaryHDU
from astropy.wcs import WCS
from reproject import reproject_interp

from .types import RectangleSize, WCSCoordinates


def getSDSSfields(
    coordinates: WCSCoordinates, size: float
) -> pd.Dataframe:  # all in degree
    """
    This function looks for the corresponding fields (run,camcol and field) in the SDSS Databank and returns a
    pandas Dataframe with the wanted fields and also some other meta information
    """
    fmt = "csv"
    default_url = "http://skyserver.sdss3.org/public/en/tools/search/x_sql.aspx"
    delta = 0.5 * size + 0.13
    ra_max = coordinates.RA + delta  # * 1.5
    ra_min = coordinates.RA - delta  # * 1.5
    dec_max = coordinates.DEC + delta
    dec_min = coordinates.DEC - delta

    querry = "SELECT fieldID, run, camCol, field, ra, dec, run, rerun FROM Field "
    querry += (
        "WHERE ra BETWEEN "
        + str(ra_min)
        + " and "
        + str(ra_max)
        + " and dec BETWEEN "
        + str(dec_min)
        + " and "
        + str(dec_max)
    )
    params = urllib.parse.urlencode({"cmd": querry, "format": fmt})
    url_opened = urllib.request.urlopen(default_url + "?%s" % params)
    lines = url_opened.readlines()
    return bytes_to_pandas(lines)


def bytes_to_pandas(list_of_bytes: List) -> pd.DataFrame:
    """
    This function turns the ByteStrings (intially from the function getSDSSfields)
    to a pandas Dataframe
    """
    joined_byte_list = b"".join(list_of_bytes)
    df = pd.read_csv(BytesIO(joined_byte_list))
    df.reset_index(inplace=True)
    headers = df.iloc[0]
    df_with_proper_headers = pd.DataFrame(df.values[1:], columns=headers)
    df_with_proper_headers = df_with_proper_headers.astype(
        {
            "run": int,
            "camCol": int,
            "field": int,
            "ra": float,
            "dec": float,
            "run1": int,
            "rerun": int,
        }
    )
    return df_with_proper_headers


def find_closest_field(
    SDSS_metadata_df: pd.DataFrame, coordinates: WCSCoordinates
) -> tuple:
    """
    This function looks for the closest ra and dec coordinates to the input coordinates
    and then gives back the corresponding meta data for this coordinate
    """
    selected_field = SDSS_metadata_df.iloc[
        (SDSS_metadata_df["ra"].sub(coordinates.RA).abs().idxmin())
        & (SDSS_metadata_df["dec"].sub(coordinates.DEC).abs().idxmin())
    ]
    run, camcol, field = selected_field.run, selected_field.camcol, selected_field.field
    return (run, camcol, field)


def getSDSSfiles(fieldInfo: Union[tuple, list], band: str, filepath: str):
    """Download SDSS file"""
    run, camcol, field = fieldInfo
    filename = f"frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
    http = f"https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/{run}/{camcol}/{filename}"

    with open(filepath, "wb") as f:
        content = requests.get(http).content
        f.write(content)


def create_sdss_image_array(
    r_band: PrimaryHDU, g_band: PrimaryHDU, b_band: PrimaryHDU
) -> np.ndarray:
    """
    This funciton combines the red, green and blue channels from fits files to a optical image.
    The mean of the channels is therefore calculated.
    """
    r = r_band.data
    g = g_band.data
    b = b_band.data
    optical_array = r / 3 + g / 3 + b / 3
    return optical_array


def create_sdss_fits_file(data: np.ndarray, filepath: str):
    hdu = PrimaryHDU(data)
    hdu = HDUList([hdu])
    hdu.writeto(filepath)


def reproject_image_array(
    hdu: PrimaryHDU,
    coordinates: WCSCoordinates,
    resolution_delt1: float,
    resolution_delt2: float,
    image_size: Union[int, RectangleSize],
) -> np.ndarray:
    """
    This function gives back a reprojection or rather a rotated array of the input image array
    based on the SIN/Orthographic projection in the World Coordinate System
    """
    if isinstance(image_size, int):
        image_size = RectangleSize(image_size, image_size)
    wcs_input_dict = {
        "CTYPE1": "RA---TAN",
        "CUNIT1": "deg",
        "CDELT1": resolution_delt1,
        "NAXIS1": 2 * image_size.image_width,
        "CRPIX1": image_size.image_width,
        "CRVAL1": coordinates.RA,
        "CTYPE2": "DEC--TAN",
        "CUNIT2": "deg",
        "CDELT2": resolution_delt2,
        "NAXIS2": 2 * image_size.image_height,
        "CRPIX2": image_size.image_height,
        "CRVAL2": coordinates.DEC,
    }
    wcs_setup = WCS(wcs_input_dict)
    reprojected_array = reproject_interp(
        input_data=hdu,
        output_projection=wcs_setup,
        shape_out=[2 * image_size.image_width, 2 * image_size.image_height],
        return_footprint=False,
    )
    return reprojected_array
