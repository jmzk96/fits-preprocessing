"""FITS file helper functions

This module contains functionality helping with the FITS file I/O.
This means reading in FITS tables, extracting coordinates and
meta information as well as creating 2D cutouts of objects of
interest.
"""
import os
from typing import NamedTuple, Optional, Union

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
    """Right ascension (RA) and declination (DEC) coordinates"""

    RA: float
    DEC: float


class RectangleSize(NamedTuple):
    """The height and width of a rectangle"""

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
    """Reads in the shimwell catalog as DataFrame

    Additionally converts the byte columns to string such that they
    can be used in filtering actions via str-functions.
    """
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


def load_mosaic(mosaic_id: str, path: str, download=False) -> Optional[PrimaryHDU]:
    """Load mosaic with mosaic_id from filepath

    This function loads the specified mosaic file with `mosaic_id`
    from the the specified path. If download is set to True, the
    function will try to download the mosaic from the
    [official LOFAR data release page](https://lofar-surveys.org/releases.html)
    and store it under the specified `path`. This requires that a valid
    mosaic_id is provided.
    """

    mosaic_filepath = create_mosaic_filepath(mosaic_id, path)

    if not os.path.exists(mosaic_filepath) and download:
        log.debug(f"Downloading {mosaic_id}")
        download_mosaic(mosaic_id=mosaic_id, path=path)

    try:
        log.debug(f"Loading {mosaic_filepath}")
        return fits.open(mosaic_filepath)[0]
    except FileNotFoundError as e:
        log.error(e)
        return None


def get_sizes_of_objects(mosaic_id, mosaic_path, catalog_path):
    """
    Get coordinates and sizes based on information in catalogue

    This function accepts a id for a mosaic as given by the FITS catalogue, it also
    accepts path to the mosaic file and the path to FITS catalogue file. The coordinates
    and the sizes of the objects are then retrieved and turned into WSCoordinates amd
    RectangleSize instances respectively. the list of WSCoordinates and list of RectangleSize
    are returned

    Parameters:
        mosaic_id (str): Mosaic ID to look for in FITS catalogue to list out its object coordinates
        and object sizes.
        mosaic_path (str): File path to FITS Mosaic file.
        catalog_path (str): File path to FITS Catalogue file.

    Returns:
        list_of_coordinates (List[WSCoordinates]): List of WSCoordinates instances of RA and DEC coordinates
        of mosaic objects.
        list_of_sizes (List[RectangleSize]): List of RectangleSize instances of Maj Sizes of objects (Maj because
        it is the biggest dimension for an ellipse)

    """
    catalog = read_shimwell_catalog(catalog_path)
    catalog = catalog[(catalog.Mosaic_ID.str.contains(mosaic_id))]
    mosaic_header = load_mosaic(mosaic_id=mosaic_id, path=mosaic_path, download=True)[
        0
    ].header
    cdelt = mosaic_header["CDELT1"]
    if cdelt < 1.0:
        cdelt = cdelt * -1.0
    # convert arcsec to degrees, then convert degrees to pixels
    catalog[["Maj", "Min", "E_Maj", "E_Min"]] = catalog[
        ["Maj", "Min", "E_Maj", "E_Min"]
    ].apply(lambda x: x * 1 / 3600 * 1 / cdelt)
    # add padding for object
    catalog["Maj"] = catalog["Maj"] + catalog["E_Maj"]
    catalog["Min"] = catalog["Min"] + catalog["E_Min"]
    list_of_coordinates = list(
        map(
            WCSCoordinates,
            np.array(catalog[["RA", "DEC"]].values.tolist())[:, 0],
            np.array(catalog[["RA", "DEC"]].values.tolist())[:, 1],
        )
    )
    list_of_sizes = list(
        map(
            RectangleSize,
            np.array(catalog[["Maj", "Maj"]].values.tolist())[:, 0],
            np.array(catalog[["Maj", "Maj"]].values.tolist())[:, 1],
        )
    )

    return list_of_coordinates, list_of_sizes


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
    return create_cutout2D(hdu, coordinates, size, wcs).data.flatten().astype("float32")


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
