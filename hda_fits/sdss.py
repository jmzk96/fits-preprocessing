import urllib
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
import requests
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.io.fits.hdu.image import PrimaryHDU

from .types import WCSCoordinates


def getSDSSfields(coordinates: WCSCoordinates, size: float):  # all in degree

    fmt = "csv"
    default_url = "http://skyserver.sdss3.org/public/en/tools/search/x_sql.aspx"
    delta = 0.5 * size + 0.13
    ra_max = coordinates.RA + 1.5 * delta
    ra_min = coordinates.RA - 1.5 * delta
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


def bytes_to_pandas(list_of_bytes: List):
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


def find_closest_field(SDSS_metadata_df: pd.DataFrame, coordinates: WCSCoordinates):
    selected_field = SDSS_metadata_df.iloc[
        (SDSS_metadata_df["ra"].sub(coordinates.ra).abs().idxmin())
        & (SDSS_metadata_df["dec"].sub(coordinates.dec).abs().idxmin())
    ]
    run, camcol, field = selected_field.run, selected_field.camcol, selected_field.field
    return (run, camcol, field)


def getSDSSfiles(fieldInfo, band, filepath):

    run = str(fieldInfo[0])
    camcol = str(fieldInfo[1])
    field = str(fieldInfo[2])

    filename = (
        "frame-" + band + "-"
        "{0:06d}".format(int(run))
        + "-"
        + camcol
        + "-"
        + "{0:04d}".format(int(field))
        + ".fits"
    )
    http = "https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/"

    http += run + "/"
    http += camcol + "/"
    http += filename
    with open(filepath, "wb") as f:
        content = requests.get(http).content
        f.write(content)


def create_sdss_image_array(r_band: PrimaryHDU, g_band: PrimaryHDU, b_band: PrimaryHDU):
    r = r_band.data
    g = g_band.data
    b = b_band.data
    optical_array = r / 3 + g / 3 + b / 3
    return optical_array


def create_sdss_fits_file(data: np.array, filepath):
    hdu = PrimaryHDU(data)
    hdu = HDUList([hdu])
    hdu.writeto(filepath)


def write_sdss_fits_header():
    pass
