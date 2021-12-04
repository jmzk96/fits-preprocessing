import urllib
from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.io.fits.hdu.image import PrimaryHDU

from .types import SDSSFields, WCSCoordinates


def get_sdss_fields(
    coordinates: WCSCoordinates, size: float = 1, number_of_entries: int = 1
) -> pd.DataFrame:  # all in degree

    fmt = "csv"
    default_url = "http://skyserver.sdss3.org/public/en/tools/search/x_sql.aspx"
    ra_max = coordinates.RA + size
    ra_min = coordinates.RA - size
    dec_max = coordinates.DEC + size
    dec_min = coordinates.DEC - size

    query = f"""
        SELECT TOP {number_of_entries} fieldID as field_id, run, camCol as cam_col, field, ra, dec, rerun,
        ra - {coordinates.RA} as delta_ra,
        dec - {coordinates.DEC} as delta_dec,
            (ra - {coordinates.RA}) * (ra - {coordinates.RA}) +
            (dec - {coordinates.DEC}) * (dec - {coordinates.DEC})
        as squared_distance
        FROM Field
        WHERE ra  BETWEEN {ra_min}  AND {ra_max}
        AND   dec BETWEEN {dec_min} AND {dec_max}
        ORDER BY (ra - {coordinates.RA}) * (ra - {coordinates.RA}) + (dec - {coordinates.DEC}) * (dec - {coordinates.DEC})
        """

    params = urllib.parse.urlencode({"cmd": query, "format": fmt})
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
            "field_id": str,
            "run": int,
            "cam_col": int,
            "field": int,
            "ra": float,
            "dec": float,
            "rerun": int,
            "delta_ra": float,
            "delta_dec": float,
            "squared_distance": float,
        }
    )
    return df_with_proper_headers


def download_sdss_file(fields: SDSSFields, band: str, filepath: Path):
    """Download SDSS file"""
    run, camcol, field = fields
    filename = f"frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
    http = f"https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/{run}/{camcol}/{filename}"

    with open(filepath, "wb") as f:
        content = requests.get(http).content
        f.write(content)


def load_sdss_files(fields: SDSSFields, path: Path, download=True):
    primary_hdus = []
    bands = ["r", "g", "i"]

    path = Path(path)
    dirpath = path / "raw" / str(fields)
    dirpath.mkdir(parents=True, exist_ok=True)

    for band in bands:
        filepath = dirpath / f"{fields}-{band}.fits"
        if not filepath.exists():
            download_sdss_file(fields=fields, band=band, filepath=filepath)
        else:
            print("Already got it")

        hdu = fits.open(filepath)[0]
        primary_hdus.append(hdu)

    return primary_hdus


def create_sdss_image_array(r_band: PrimaryHDU, g_band: PrimaryHDU, b_band: PrimaryHDU):
    r = r_band.data
    g = g_band.data
    b = b_band.data
    optical_array = r / 3 + g / 3 + b / 3
    return optical_array


def create_sdss_fits_file(data: np.ndarray, filepath):
    hdu = PrimaryHDU(data)
    hdu = HDUList([hdu])
    hdu.writeto(filepath)
