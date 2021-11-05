"""PINK related helper functions

This module provides I/O functionality related to the PINK self
organizing maps application. 
"""
import struct
from typing import List, Union

import numpy as np
import numpy.typing as npt
from astropy.io.fits.hdu.image import PrimaryHDU

import hda_fits.fits as hfits
from hda_fits.fits import RectangleSize, WCSCoordinates
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
    min_max_scale: bool = True,
) -> str:
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
        data = hfits.create_cutout2D_as_flattened_numpy_array(hdu, coord, image_size)

        if min_max_scale:
            dmax, dmin = data.max(), data.min()
            data = (data - dmin) / (dmax - dmin)

        if data.size == number_of_pixels:
            write_pink_file_v2_data(filepath, data)
        else:
            log.warning(
                f"Data was truncated. Expected {number_of_pixels}, got {data.size} floats."
            )
            number_of_images -= 1
            write_pink_file_v2_header(
                filepath=filepath,
                number_of_images=number_of_images,
                image_height=image_size.image_height,
                image_width=image_size.image_width,
                overwrite=True,
            )

    log.debug(f"Wrote {number_of_images} images to {filepath}")

    return number_of_images

def write_all_objects_pink_file_v2(
    catalog_path: str,
    filepath: str,
    output_mosaic_path: str,
    image_size: Union[int, RectangleSize],
    min_max_scale: bool = True,
    save_in_different_files : bool = True,
    download: bool = False,
):
    catalog = hda_fits.read_shimwell_catalog(catalog_path,reduced=True)
    list_of_mosaics=catalog.Mosaic_ID.unique().tolist()
    number_of_images = 0
    for i in list_of_mosaics:
        mosaic = hda_fits.load_mosaic(i, output_mosaic_path, download = download)
        hdu = mosaic[0]
        table_with_unique_mosaic = catalog[catalog.Mosaic_ID == i]
        coord = table_with_unique_mosaic.loc[:,["RA","DEC"]].values.tolist()
        if save_in_different_files:
            pink_bin_file = hda_fits.write_mosaic_objects_to_pink_file_v2(filepath=filepath + f"{i}.bin",
                                                                          coordinates=coord,hdu=hdu,
                                                                          image_size=image_size, 
                                                                          min_max_scale=min_max_scale)
        else:
            number_of_images += hda_fits.write_mosaic_objects_to_pink_file_v2(filepath=filepath + "all_objects_pink.bin",
                                                                          coordinates=coord,hdu=hdu,
                                                                          image_size=image_size, 
                                                                          min_max_scale=min_max_scale)
            write_pink_file_v2_header(filepath=filepath + "all_objects_pink.bin",number_of_images=number_of_images,
                                                                                image_height=image_size.image_height,
                                                                                image_width=image_size.image_width,
                                                                                overwrite=True)
                                                                        