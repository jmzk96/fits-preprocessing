import pytest
import os

import hda_fits as hfits
from hda_fits.fits import RectangleSize, WCSCoordinates
from hda_fits.logging_config import logging


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def test_write_square_mosaic_cutout_to_pink_file(
    tmp_path,
    mosaic_hdu_and_wcs,
    example_object_world_coordinates,
):
    c1 = example_object_world_coordinates

    hdu, _ = mosaic_hdu_and_wcs
    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    number_of_written_images = hfits.write_mosaic_objects_to_pink_file_v2(
        filepath=tmp_filepath,
        hdu=hdu,
        coordinates=[c1, c1, c1],
        image_size=20,
    )

    assert number_of_written_images == 3
    assert tmp_filepath.exists()


def test_write_rectangular_mosaic_cutout_to_pink_file(
    tmp_path,
    mosaic_hdu_and_wcs,
    example_object_world_coordinates,
):

    c1 = example_object_world_coordinates

    hdu, wcs = mosaic_hdu_and_wcs
    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    number_of_written_images = hfits.write_mosaic_objects_to_pink_file_v2(
        filepath=tmp_filepath,
        hdu=hdu,
        coordinates=[c1, c1, c1],
        image_size=RectangleSize(image_height=20, image_width=10),
    )

    assert number_of_written_images == 3
    assert os.path.exists(tmp_filepath)


def test_write_mosaic_cutout_to_pink_file_coordinates_outside_of_mosaic(
    tmp_path,
    mosaic_hdu_and_wcs,
    example_object_world_coordinates_outside,
):

    hdu, wcs = mosaic_hdu_and_wcs
    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    number_of_written_images = hfits.write_mosaic_objects_to_pink_file_v2(
        filepath=tmp_filepath,
        hdu=hdu,
        coordinates=[example_object_world_coordinates_outside],
        image_size=20,
    )

    assert number_of_written_images == 0
    assert os.path.exists(tmp_filepath)
