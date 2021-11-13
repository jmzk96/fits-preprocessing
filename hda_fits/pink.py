"""PINK related helper functions

This module provides I/O functionality related to the PINK self
organizing maps application.
"""
import struct
from typing import List, Union

import numpy as np
import pandas as pd
from astropy.io.fits.hdu.image import PrimaryHDU

import hda_fits.fits as hfits
import hda_fits.pink as hpink
from hda_fits.fits import RectangleSize, WCSCoordinates, load_mosaic
from hda_fits.logging_config import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def write_pink_file_v2_header(
    filepath: str,
    number_of_images: int,
    image_height: int,
    image_width: int,
    overwrite: bool = False,
):
    with open(filepath, "r+b" if overwrite else "wb") as f:
        f.write(
            struct.pack(
                "i" * 8, 2, 0, 0, number_of_images, 0, 2, image_height, image_width
            )
        )


def write_pink_file_v2_data(filepath, data: np.ndarray):
    with open(filepath, "ab") as f:
        f.write(struct.pack("%sf" % data.size, *data.tolist()))


def write_mosaic_objects_to_pink_file_v2(
    filepath: str,
    hdu: PrimaryHDU,
    coordinates: List[WCSCoordinates],
    image_size: Union[int, RectangleSize],
    min_max_scale: bool = False,
    fill_nan=False,
) -> int:
    if isinstance(image_size, int):
        image_size = RectangleSize(image_size, image_size)

    number_of_pixels = image_size.image_height * image_size.image_width
    number_of_images = len(coordinates)

    write_pink_file_v2_header(
        filepath=filepath,
        number_of_images=number_of_images,
        image_height=image_size.image_height,
        image_width=image_size.image_width,
        overwrite=False,
    )
    for coord in coordinates:
        try:
            data = hfits.create_cutout2D_as_flattened_numpy_array(
                hdu, coord, image_size
            )
            if np.isnan(data).any():
                if fill_nan:
                    data = np.nan_to_num(data, 0.0)
                else:
                    raise ValueError("Objects data array contains NaNs")
        except ValueError as e:
            log.warning(e)
            log.warning(f"Image at coordinates {coord} not added to pink file")
            number_of_images -= 1
            continue

        if min_max_scale:
            dmax, dmin = data.max(), data.min()
            data = (data - dmin) / (dmax - dmin)

        if data.size == number_of_pixels:
            write_pink_file_v2_data(filepath, data)
        else:
            log.warning(
                f"Data was truncated. Expected {number_of_pixels}, got {data.size} floats."
            )
            log.warning(f"Image at coordinates {coord} not added to pink file")
            number_of_images -= 1
            write_pink_file_v2_header(
                filepath=filepath,
                number_of_images=number_of_images,
                image_height=image_size.image_height,
                image_width=image_size.image_width,
                overwrite=True,
            )

    log.info(f"Wrote {number_of_images} images to {filepath}")

    return number_of_images


def write_all_objects_pink_file_v2(
    catalog_path: str,
    filepath: str,
    mosaic_path: str,
    image_size: Union[int, RectangleSize],
    min_max_scale: bool = True,
    save_in_different_files: bool = True,
    download: bool = False,
):
    if isinstance(image_size, int):
        image_size = RectangleSize(image_height=image_size, image_width=image_size)

    catalog = hfits.read_shimwell_catalog(catalog_path, reduced=True)
    list_of_mosaics = catalog.Mosaic_ID.unique().tolist()
    number_of_images = 0
    for i in list_of_mosaics:
        hdu = hfits.load_mosaic(i, mosaic_path, download=download)
        table_with_unique_mosaic = catalog[catalog.Mosaic_ID == i]
        coord = table_with_unique_mosaic.loc[:, ["RA", "DEC"]].values.tolist()
        if save_in_different_files:
            hpink.write_mosaic_objects_to_pink_file_v2(
                filepath=filepath + f"{i}.bin",
                coordinates=coord,
                hdu=hdu,
                image_size=image_size,
                min_max_scale=min_max_scale,
            )
        else:
            number_of_images += hpink.write_mosaic_objects_to_pink_file_v2(
                filepath=filepath + "all_objects_pink.bin",
                coordinates=coord,
                hdu=hdu,
                image_size=image_size,
                min_max_scale=min_max_scale,
            )
            write_pink_file_v2_header(
                filepath=filepath + "all_objects_pink.bin",
                number_of_images=number_of_images,
                image_height=image_size.image_height,
                image_width=image_size.image_width,
                overwrite=True,
            )


def write_catalog_objects_pink_file_v2(
    filepath: str,
    catalog: pd.DataFrame,
    mosaic_path: str,
    image_size: Union[int, RectangleSize],
    min_max_scale: bool = False,
    download: bool = False,
    fill_nan: bool = False,
):
    """
    Writes all images in a given catalog to a binary file in PINK v2 format.
    This includes loading (and optionally downloading) each required mosaic
    and updating the sum of written images at the end.
    """

    if isinstance(image_size, int):
        image_size = RectangleSize(image_height=image_size, image_width=image_size)

    mosaic_ids = catalog["Mosaic_ID"].unique().tolist()

    number_of_images = 0

    for mosaic_id in mosaic_ids:
        hdu = load_mosaic(mosaic_id=mosaic_id, path=mosaic_path, download=download)

        coordinates = catalog[catalog["Mosaic_ID"] == mosaic_id][
            ["RA", "DEC"]
        ].values.tolist()

        number_of_images_current = write_mosaic_objects_to_pink_file_v2(
            filepath=filepath,
            hdu=hdu,
            coordinates=coordinates,
            image_size=image_size,
            min_max_scale=min_max_scale,
            fill_nan=fill_nan,
        )

        number_of_images += number_of_images_current

    write_pink_file_v2_header(
        filepath=filepath,
        number_of_images=number_of_images,
        image_height=image_size.image_height,
        image_width=image_size.image_width,
        overwrite=True,
    )

    return number_of_images
