"""FITS file helper functions

This module contains functionality helping with the FITS file I/O.
This means reading in FITS tables, extracting coordinates and
meta information as well as creating 2D cutouts of objects of
interest.
"""
import os
from typing import Tuple
import requests
import pandas as pd
from hda_fits.logging_config import logging
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.io.fits.hdu.image import PrimaryHDU

from typing import Tuple, Union, NamedTuple


class WCSCoordinates(NamedTuple):
    RA: float
    DEC: float


class RectangleSize(NamedTuple):
    width: int
    height: int


log = logging.getLogger(__name__)


MOSAIC_FILENAME_TEMPLATE = "{}-mosaic.fits"


def column_dtype_byte_to_string(df: pd.DataFrame):
    byte_cols = df.select_dtypes(include=object).columns
    df[byte_cols] = df[byte_cols].apply(lambda col: col.str.decode("utf-8"))
    return df


def read_shimwell_catalog(path: str, reduced=False):
    table = Table.read(path).to_pandas()
    if reduced:
        table = table.loc[:, "Source_Name", "RA", "DEC", "Mosaic_ID"].copy()
    return column_dtype_byte_to_string(table)


def create_mosaic_filepath(mosaic_id: str, path: str):
    mosaic_filename = MOSAIC_FILENAME_TEMPLATE.format(mosaic_id)
    return os.path.join(path, mosaic_filename)


def download_mosaic(mosaic_id: str, path: str = ""):
    mosaic_filename = MOSAIC_FILENAME_TEMPLATE.format(mosaic_id)
    mosaic_filepath = os.path.join(path, mosaic_filename)

    mosaic_url = f"https://lofar-surveys.org/public/mosaics/{mosaic_filename}"

    with requests.get(mosaic_url, stream=True) as r:
        r.raise_for_status()
        with open(mosaic_filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return mosaic_filepath


def load_mosaic(mosaic_id: str, path: str, download=False):
    mosaic_filepath = create_mosaic_filepath(mosaic_id, path)

    if not os.path.exists(mosaic_filepath) and download:
        log.debug(f"Downloading {mosaic_id}")
        download_mosaic(mosaic_id=mosaic_id, path=path)

    try:
        log.debug(f"Loading {mosaic_filepath}")
        return fits.open(mosaic_filepath)
    except FileNotFoundError as e:
        log.error(e)


def create_cutout2D(
    hdu: PrimaryHDU, coordinates: WCSCoordinates, cutout_size: Union[int, RectangleSize]
):
    wcs = WCS(hdu.header)
    position = wcs.wcs_world2pix([coordinates], 0)
    return Cutout2D(hdu.data, position[0], cutout_size)
