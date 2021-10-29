"""FITS file helper functions

This module contains functionality helping with the FITS file I/O.
This means reading in FITS tables, extracting coordinates and
meta information as well as creating 2D cutouts of objects of
interest.
"""
import os
from typing import List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.io.fits.hdu.image import PrimaryHDU
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS

from hda_fits.logging_config import logging


class WCSCoordinates(NamedTuple):
    RA: float
    DEC: float


class RectangleSize(NamedTuple):
    image_height: int
    image_width: int


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

MOSAIC_FILENAME_TEMPLATE = "{}-mosaic.fits"


def column_dtype_byte_to_string(df: pd.DataFrame) -> pd.DataFrame:
    byte_cols = df.select_dtypes(include=object).columns
    df[byte_cols] = df[byte_cols].apply(lambda col: col.str.decode("utf-8"))
    return df


def read_shimwell_catalog(path: str, reduced=False) -> pd.DataFrame:
    table = Table.read(path).to_pandas()
    if reduced:
        table = table.loc[:, ["Source_Name", "RA", "DEC", "Mosaic_ID"]].copy()
    return column_dtype_byte_to_string(table)


def create_mosaic_filepath(mosaic_id: str, path: str) -> str:
    mosaic_filename = MOSAIC_FILENAME_TEMPLATE.format(mosaic_id)
    return os.path.join(path, mosaic_filename)


def download_mosaic(mosaic_id: str, path: str = "") -> str:
    mosaic_filename = MOSAIC_FILENAME_TEMPLATE.format(mosaic_id)
    mosaic_filepath = os.path.join(path, mosaic_filename)

    mosaic_url = f"https://lofar-surveys.org/public/mosaics/{mosaic_filename}"

    with requests.get(mosaic_url, stream=True) as r:
        r.raise_for_status()
        with open(mosaic_filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return mosaic_filepath


def load_mosaic(
    mosaic_id: str, path: str, download=False
) -> Optional[List[PrimaryHDU]]:
    mosaic_filepath = create_mosaic_filepath(mosaic_id, path)

    if not os.path.exists(mosaic_filepath) and download:
        log.debug(f"Downloading {mosaic_id}")
        download_mosaic(mosaic_id=mosaic_id, path=path)

    try:
        log.debug(f"Loading {mosaic_filepath}")
        return fits.open(mosaic_filepath)
    except FileNotFoundError as e:
        log.error(e)
        return None


def create_cutout2D(
    hdu: PrimaryHDU,
    coordinates: WCSCoordinates,
    size: Union[int, RectangleSize],
    wcs: WCS = None,
) -> Cutout2D:
    if not wcs:
        wcs = WCS(hdu.header)
    position = wcs.wcs_world2pix([coordinates], 0)
    cutout = Cutout2D(hdu.data, position[0], size, wcs=wcs)
    return cutout


def create_cutout2D_as_flattened_numpy_array(
    hdu: PrimaryHDU,
    coordinates: WCSCoordinates,
    size: Union[int, RectangleSize],
    wcs: WCS = None,
) -> np.ndarray:
    return create_cutout2D(hdu, coordinates, size, wcs).data.flatten()


def create_cutout2D_as_updated_hdu(
    hdu: PrimaryHDU,
    coordinates: WCSCoordinates,
    size: Union[int, RectangleSize],
    wcs: WCS = None,
) -> PrimaryHDU:
    cutout = create_cutout2D(hdu, coordinates, size, wcs)
    hdu_cutout = hdu.copy()
    hdu_cutout.data = cutout.data
    hdu_cutout.header.update(cutout.wcs.to_header())
    return hdu_cutout
