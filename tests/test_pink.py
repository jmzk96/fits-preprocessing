import os
import struct

import hda_fits as hfits
from hda_fits import fits, pink
from hda_fits.logging_config import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def test_write_pink_file_v2_header(
    tmp_path,
    pink_bin_header,
    number_of_images=3,
    image_height=200,
    image_width=200,
):
    tmp_filepath = tmp_path / pink_bin_header

    pink.write_pink_file_v2_header(
        tmp_filepath, number_of_images, image_height, image_width, overwrite=False
    )
    with open(tmp_filepath, "r+b") as g:
        content = g.read()
        assert content == struct.pack(
            "i" * 8, 2, 0, 0, number_of_images, 0, 2, 200, 200
        )

    new_number_of_images = number_of_images + 2

    pink.write_pink_file_v2_header(
        tmp_filepath, new_number_of_images, image_height, image_width, overwrite=True
    )
    with open(tmp_filepath, "r+b") as g:
        content = g.read()
        assert content == struct.pack(
            "i" * 8, 2, 0, 0, new_number_of_images, 0, 2, 200, 200
        )


def test_write_pink_file_v2_data(
    tmp_path, pink_bin_data, mosaic_hdu_and_wcs, example_object_world_coordinates
):
    hdu, wcs = mosaic_hdu_and_wcs
    example_array = fits.create_cutout2D_as_flattened_numpy_array(
        hdu, example_object_world_coordinates, 20, wcs
    )

    tmp_filepath = tmp_path / pink_bin_data

    pink.write_pink_file_v2_data(tmp_filepath, example_array)
    with open(tmp_filepath, "r+b") as g:
        content = g.read()
        assert content == struct.pack(
            "%sf" % example_array.size, *example_array.tolist()
        )


def test_write_mosaic_objects_to_pink_file_v2_with_square_image_size(
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


def test_write_mosaic_objects_to_pink_file_v2_with_rectangular_image_size(
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
        image_size=fits.RectangleSize(image_height=20, image_width=10),
    )

    assert number_of_written_images == 3
    assert os.path.exists(tmp_filepath)


def test_write_mosaic_objects_to_pink_file_v2_with_coordinate_outside_of_mosaic(
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
