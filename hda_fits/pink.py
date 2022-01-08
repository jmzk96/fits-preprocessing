"""PINK related helper functions

This module provides I/O functionality related to the PINK self
organizing maps application.
"""
import struct
from typing import BinaryIO, List, Tuple, Union

import numpy as np
import pandas as pd
from astropy.io.fits.hdu.image import PrimaryHDU

import hda_fits.fits as hfits
from hda_fits import image_processing as himg
from hda_fits import panstarrs as ps
from hda_fits.fits import RectangleSize, WCSCoordinates, load_mosaic
from hda_fits.logging_config import logging
from hda_fits.sdss import (
    create_reprojected_rgb_image,
    extract_crossmatch_attributes,
    load_sdss_field_files,
)
from hda_fits.types import Layout, PinkHeader

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def read_pink_file_header_from_stream(file_stream: BinaryIO) -> PinkHeader:
    (
        version,
        file_type,
        data_type,
        number_of_images,
        data_layout,
        dimensionality,
    ) = struct.unpack("i" * 6, file_stream.read(4 * 6))

    depth = struct.unpack("i", file_stream.read(4))[0] if dimensionality > 2 else 1
    height = struct.unpack("i", file_stream.read(4))[0] if dimensionality > 1 else 1
    width = struct.unpack("i", file_stream.read(4))[0]

    pink_layout = Layout(width=width, height=height, depth=depth)

    header_end_offset = file_stream.tell()

    return PinkHeader(
        version=version,
        file_type=file_type,
        data_type=data_type,
        number_of_images=number_of_images,
        data_layout=data_layout,
        dimensionality=dimensionality,
        layout=pink_layout,
        header_end_offset=header_end_offset,
    )


def read_pink_file_header(filepath: str) -> PinkHeader:
    with open(filepath, "rb") as file_stream:
        header = read_pink_file_header_from_stream(file_stream=file_stream)

    return header


def read_pink_file_image_from_stream(
    file_stream: BinaryIO,
    image_number: int,
    header_offset: int,
    layout: Layout,
):
    """Read an image from an absolute position

    This function will seek to the absolute position in the open file
    and read the `image_size` floating point values.
    """
    image_size = layout.width * layout.height * layout.depth
    file_stream.seek(image_size * image_number * 4 + header_offset, 0)
    data = struct.unpack("f" * image_size, file_stream.read(image_size * 4))

    if layout.depth == 1:
        image = np.array(data).reshape((layout.height, layout.width))
    else:
        image = np.array(data).reshape((layout.depth, layout.height, layout.width))

    return image


def read_pink_file_image(filepath: str, image_number: int) -> np.ndarray:
    """Reads and reshapes the data of a .pink image"""
    with open(filepath, "rb") as file_stream:
        header = read_pink_file_header_from_stream(file_stream=file_stream)
        layout = header.layout
        image_size = layout.width * layout.height * layout.depth

        file_stream.seek(image_number * image_size * 4, 1)
        data = struct.unpack("f" * image_size, file_stream.read(image_size * 4))

        if layout.depth == 1:
            image = np.array(data).reshape((layout.height, layout.width))
        else:
            image = np.array(data).reshape((layout.depth, layout.height, layout.width))

    return image


def read_pink_file_multiple_images(
    filepath: str, image_numbers: List[int]
) -> List[np.ndarray]:
    """Reads and reshapes multiple .pink images"""

    images = []

    with open(filepath, "rb") as file_stream:
        header = read_pink_file_header_from_stream(file_stream=file_stream)
        layout = header.layout

        for image_number in image_numbers:
            image = read_pink_file_image_from_stream(
                file_stream,
                image_number=image_number,
                header_offset=header.header_end_offset,
                layout=layout,
            )
            images.append(image)

    return images


def write_pink_file_header(
    filepath: str,
    number_of_images: int,
    image_height: int,
    image_width: int,
    overwrite: bool = False,
    version: str = "v2",
):
    if version == "v2":
        with open(filepath, "r+b" if overwrite else "wb") as f:
            f.write(
                struct.pack(
                    "i" * 8, 2, 0, 0, number_of_images, 0, 2, image_height, image_width
                )
            )
    elif version == "v1":
        with open(filepath, "r+b" if overwrite else "wb") as f:
            f.write(
                struct.pack("i" * 4, number_of_images, 1, image_width, image_height)
            )


def write_pink_file_header_multichannel(
    filepath: str,
    number_of_images: int,
    image_layout: Layout,
    overwrite: bool = False,
):

    width, height, depth = image_layout

    with open(filepath, "r+b" if overwrite else "wb") as f:
        f.write(
            struct.pack("i" * 9, 2, 0, 0, number_of_images, 0, 3, depth, height, width)
        )


def convert_pink_file_header_v1_to_v2(filepath: str):
    with open(filepath, "rb") as file_stream:
        (
            number_of_images,
            number_of_channels,
            image_width,
            image_height,
        ) = struct.unpack("i" * 4, file_stream.read(4 * 4))
        try:
            with open(filepath, "rb") as file_stream:
                len_of_file = len(file_stream.read())
                file_stream.seek(16)
                data = file_stream.read()
                log.info(f"Extracted {(len_of_file-16)/4} floats from file_stream")
        except ValueError as e:
            log.warning(e)
            log.warning("No trailing data after header or wrong file_stream format")

        write_pink_file_header(
            filepath=filepath,
            number_of_images=number_of_images,
            image_height=image_height,
            image_width=image_width,
            overwrite=False,
            version="v2",
        )
        try:
            with open(filepath, "ab") as file_stream:
                file_stream.write(data)
        except ValueError as e:
            log.warning(e)
            log.warning("No trailing data after header")


def convert_pink_file_header_v2_to_v1(filepath: str):
    with open(filepath, "rb") as file_stream:
        list_of_parameters = struct.unpack("i" * 8, file_stream.read(4 * 8))
        number_of_images, image_height, image_width = (
            list_of_parameters[3],
            list_of_parameters[6],
            list_of_parameters[7],
        )
        try:
            with open(filepath, "rb") as file_stream:
                len_of_file = len(file_stream.read())
                file_stream.seek(32)
                data = file_stream.read()
                log.info(f"Extracted {(len_of_file-32)/4} floats from file_stream")
        except ValueError as e:
            log.warning(e)
            log.warning("No trailing data after header or wrong file_stream format")
        write_pink_file_header(
            filepath=filepath,
            number_of_images=number_of_images,
            image_height=image_height,
            image_width=image_width,
            overwrite=False,
            version="v1",
        )
        try:
            with open(filepath, "ab") as file_stream:
                file_stream.write(data)
        except ValueError as e:
            log.warning(e)
            log.warning("No trailing data after header")


def write_pink_file_v2_data(filepath: str, data: np.ndarray):
    with open(filepath, "ab") as f:
        f.write(struct.pack("%sf" % data.size, *data.tolist()))


def write_mosaic_objects_to_pink_file_v2(
    filepath: str,
    hdu: PrimaryHDU,
    coordinates: List[WCSCoordinates],
    image_size: Union[int, RectangleSize],
    min_max_scale: bool = False,
    denoise: bool = True,
    fill_nan=False,
    overwrite_header=False,
) -> List[bool]:

    if isinstance(image_size, int):
        image_size = RectangleSize(image_size, image_size)

    number_of_pixels = image_size.image_height * image_size.image_width

    number_of_images = len(coordinates)
    image_was_written = []

    write_pink_file_header(
        filepath=filepath,
        number_of_images=number_of_images,
        image_height=image_size.image_height,
        image_width=image_size.image_width,
        overwrite=overwrite_header,
    )
    for coord in coordinates:
        try:
            data = hfits.create_cutout2D_as_flattened_numpy_array(
                hdu, coord, image_size
            )
            if np.isnan(data).any():
                if fill_nan:
                    data = np.nan_to_num(data, data.mean())
                else:
                    raise ValueError("Objects data array contains NaNs")

            if denoise:
                data = himg.denoise_cutouts_from_mean(data)

        except ValueError as e:
            log.warning(e)
            log.warning(f"Image at coordinates {coord} not added to pink file_stream")
            image_was_written.append(False)
            continue

        if min_max_scale:
            dmax, dmin = data.max(), data.min()
            data = (data - dmin) / (dmax - dmin)

        if data.size == number_of_pixels:
            write_pink_file_v2_data(filepath, data)
            image_was_written.append(True)
        else:
            log.warning(
                f"Data was truncated. Expected {number_of_pixels}, got {data.size} floats."
            )
            log.warning(f"Image at coordinates {coord} not added to pink file_stream")
            image_was_written.append(False)

    number_of_images = sum(image_was_written)
    write_pink_file_header(
        filepath=filepath,
        number_of_images=number_of_images,
        image_height=image_size.image_height,
        image_width=image_size.image_width,
        overwrite=True,
    )

    log.info(f"Wrote {number_of_images} images to {filepath}")

    return image_was_written


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
            write_mosaic_objects_to_pink_file_v2(
                filepath=filepath + f"{i}.bin",
                coordinates=coord,
                hdu=hdu,
                image_size=image_size,
                min_max_scale=min_max_scale,
            )
        else:
            image_was_written = write_mosaic_objects_to_pink_file_v2(
                filepath=filepath + "all_objects_pink.bin",
                coordinates=coord,
                hdu=hdu,
                image_size=image_size,
                min_max_scale=min_max_scale,
            )
            number_of_images += sum(image_was_written)

            write_pink_file_header(
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
    denoise: bool = True,
    download: bool = False,
    fill_nan: bool = False,
) -> pd.DataFrame:
    """
    Writes all images in a given catalog to a binary file_stream in PINK v2 format.
    This includes loading (and optionally downloading) each required mosaic
    and updating the sum of written images at the end.
    """

    if isinstance(image_size, int):
        image_size = RectangleSize(image_height=image_size, image_width=image_size)

    catalog_of_written_images = pd.DataFrame()

    mosaic_ids = catalog["Mosaic_ID"].unique().tolist()
    number_of_images_to_write = catalog.shape[0]

    log.info(f"Going to write {number_of_images_to_write} images")

    write_pink_file_header(
        filepath=filepath,
        number_of_images=number_of_images_to_write,
        image_height=image_size.image_height,
        image_width=image_size.image_width,
        overwrite=False,
    )

    number_of_images = 0

    for mosaic_id in mosaic_ids:
        hdu = load_mosaic(mosaic_id=mosaic_id, path=mosaic_path, download=download)

        catalog_mosaic_subset = catalog[catalog["Mosaic_ID"] == mosaic_id].copy()
        coordinates = catalog_mosaic_subset[["RA", "DEC"]].values.tolist()

        image_was_written = write_mosaic_objects_to_pink_file_v2(
            filepath=filepath,
            hdu=hdu,
            coordinates=coordinates,
            image_size=image_size,
            min_max_scale=min_max_scale,
            denoise=denoise,
            fill_nan=fill_nan,
            overwrite_header=True,
        )

        catalog_mosaic_subset_written = catalog_mosaic_subset[image_was_written]
        catalog_of_written_images = catalog_of_written_images.append(
            catalog_mosaic_subset_written
        )

        number_of_images += sum(image_was_written)

    write_pink_file_header(
        filepath=filepath,
        number_of_images=number_of_images,
        image_height=image_size.image_height,
        image_width=image_size.image_width,
        overwrite=True,
    )

    log.info(f"Wrote {number_of_images} images to {filepath}.")
    return catalog_of_written_images


def write_crossmatch_catalog_to_pink_file(
    crossmatch_catalog: pd.DataFrame,
    filepath: str,
    sdss_data_path: str,
    image_size: Union[int, RectangleSize],
    download: bool = False,
    fill_nan: bool = False,
):

    if isinstance(image_size, int):
        image_size = RectangleSize(image_height=image_size, image_width=image_size)

    image_height, image_width = image_size
    crossmatch_attributes = extract_crossmatch_attributes(crossmatch_catalog)
    number_of_images = len(crossmatch_attributes)

    write_pink_file_header(
        filepath,
        number_of_images=number_of_images,
        image_height=image_height,
        image_width=image_width,
    )

    images_written = np.full(len(crossmatch_attributes), True)

    assert images_written.size == crossmatch_catalog.shape[0]

    for i, cma in enumerate(crossmatch_attributes):
        _, coordinates, fields = cma.values()
        primary_hdus = load_sdss_field_files(fields, sdss_data_path, download)
        rgb_image = create_reprojected_rgb_image(
            primary_hdus=primary_hdus,
            coordinates=coordinates,
            image_size=image_size,
            merge=True,
            use_lupton_algorithm=False,
        )

        if np.isnan(rgb_image).any():
            if fill_nan:
                rgb_image = np.nan_to_num(rgb_image, rgb_image.mean())
            else:
                images_written[i] = False
                continue

        write_pink_file_v2_data(filepath=filepath, data=rgb_image.flatten())

    write_pink_file_header(
        filepath,
        number_of_images=images_written.sum(),
        image_height=image_height,
        image_width=image_width,
        overwrite=True,
    )

    return crossmatch_catalog[images_written]


def write_panstarrs_objects_to_pink_file(
    panstarrs_catalog: pd.DataFrame,
    filepath: str,
    panstarrs_data_path: str,
    image_size: Union[int, RectangleSize],
    download: bool = False,
):
    if isinstance(image_size, int):
        image_size = RectangleSize(image_height=image_size, image_width=image_size)

    image_height, image_width = image_size
    number_of_images = len(panstarrs_catalog)

    write_pink_file_header(
        filepath,
        number_of_images=number_of_images,
        image_height=image_height,
        image_width=image_width,
    )
    images_written = np.full(len(panstarrs_catalog), True)

    assert images_written.size == panstarrs_catalog.shape[0]

    list_of_source = panstarrs_catalog.Source_Name.tolist()
    list_of_ra = panstarrs_catalog.RA.tolist()
    list_of_dec = panstarrs_catalog.DEC.tolist()

    for i, (source, ra, dec) in enumerate(zip(list_of_source, list_of_ra, list_of_dec)):
        try:
            primary_hdus = ps.load_panstarrs_file(
                panstarrs_catalog, source, panstarrs_data_path, download
            )

            rgb_image = create_reprojected_rgb_image(
                primary_hdus=primary_hdus,
                coordinates=WCSCoordinates(ra, dec),
                image_size=image_size,
                merge=True,
                use_lupton_algorithm=False,
            )
            if np.isnan(rgb_image).any():
                images_written[i] = False
                continue
            write_pink_file_v2_data(filepath=filepath, data=rgb_image.flatten())
        except Exception as e:
            log.debug(e)
            images_written[i] = False

    write_pink_file_header(
        filepath,
        number_of_images=images_written.sum(),
        image_height=image_height,
        image_width=image_width,
        overwrite=True,
    )
    return panstarrs_catalog[images_written]


def transform_multichannel_images(
    image_radio: np.ndarray,
    image_optical: np.ndarray,
    channel_weights: List[float] = [1.0, 1.0],
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    # Transformations
    image_optical_masked = himg.create_and_apply_image_mask(
        image_for_mask_creation=image_optical,
        image_to_be_masked=image_radio,
        factor_std=5,
        padding=5,
        border_proportion=0.15,
        fill_with=np.mean,
    )

    image_data_optical = himg.denoise_cutouts_from_mean(image_optical_masked.flatten())
    image_data_optical = himg.min_max(image_data_optical)

    image_data_radio = image_radio.flatten()

    image_data_radio /= image_data_radio.sum() * channel_weights[0]
    image_data_optical /= image_data_optical.sum() * channel_weights[1]

    if not np.isfinite(image_data_optical).all():
        log.warning("INF or NaN in optical image. Skipping..")
        raise ValueError("NAN or INF")

    if not np.isfinite(image_data_radio).all():
        log.warning("INF or NaN in radio image. Skipping..")
        raise ValueError("NAN or INF")

    return image_data_radio, image_data_optical


def write_multichannel_pink_file(
    filepath_pink_output: str,
    filepath_pink_radio: str,
    filepath_pink_optical: str,
    transformation_function=transform_multichannel_images,
) -> np.ndarray:
    header_radio = read_pink_file_header(filepath_pink_radio)
    header_optical = read_pink_file_header(filepath_pink_optical)

    layout = Layout(
        depth=2, height=header_radio.layout.height, width=header_radio.layout.width
    )

    assert header_radio == header_optical

    # Header schreiben
    write_pink_file_header_multichannel(
        filepath=filepath_pink_output,
        number_of_images=header_radio.number_of_images,
        image_layout=layout,
    )

    images_written = np.full(header_radio.number_of_images, True)

    # Daten lesen und Daten schreiben
    for i in range(header_radio.number_of_images):
        image_radio = read_pink_file_image(filepath_pink_radio, i)
        image_optical = read_pink_file_image(filepath_pink_optical, i)

        try:
            transform_result = transformation_function(image_radio, image_optical)
        except (ValueError, Exception) as e:
            log.warning(e)
            images_written[i] = False
            continue

        image_data_radio, image_data_optical = transform_result

        data = np.concatenate([image_data_radio, image_data_optical])
        write_pink_file_v2_data(filepath=filepath_pink_output, data=data)

    write_pink_file_header_multichannel(
        filepath=filepath_pink_output,
        number_of_images=images_written.sum(),
        image_layout=layout,
        overwrite=True,
    )

    return images_written


def write_pink_subset_file(
    filepath_output_pink: str, filepath_input_pink: str, image_indices: List[int]
):
    header = read_pink_file_header(filepath_input_pink)
    width, height, _ = header.layout

    write_pink_file_header(
        filepath=filepath_output_pink,
        number_of_images=len(image_indices),
        image_height=height,
        image_width=width,
        overwrite=False,
    )

    for idx in image_indices:
        image = read_pink_file_image(filepath_input_pink, idx)
        write_pink_file_v2_data(filepath_output_pink, image.flatten())
