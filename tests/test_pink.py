import os
import struct

import numpy as np

import hda_fits as hfits
from hda_fits import fits, pink
from hda_fits.logging_config import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def test_write_pink_file_header(
    tmp_path,
    pink_bin_header,
    number_of_images=3,
    image_height=200,
    image_width=200,
):
    tmp_filepath = tmp_path / pink_bin_header
    new_number_of_images = number_of_images + 2

    pink.write_pink_file_header(
        tmp_filepath, number_of_images, image_height, image_width, overwrite=False
    )
    with open(tmp_filepath, "r+b") as g:
        content = g.read()
        assert content == struct.pack(
            "i" * 8, 2, 0, 0, number_of_images, 0, 2, 200, 200
        )

    pink.write_pink_file_header(
        tmp_filepath, new_number_of_images, image_height, image_width, overwrite=True
    )
    with open(tmp_filepath, "r+b") as g:
        content = g.read()
        assert content == struct.pack(
            "i" * 8, 2, 0, 0, new_number_of_images, 0, 2, 200, 200
        )

    pink.write_pink_file_header(
        tmp_filepath,
        number_of_images,
        image_height,
        image_width,
        overwrite=False,
        version="v1",
    )
    with open(tmp_filepath, "r+b") as g:
        content = g.read()
        assert content == struct.pack("i" * 4, number_of_images, 1, 200, 200)

    pink.write_pink_file_header(
        tmp_filepath,
        new_number_of_images,
        image_height,
        image_width,
        overwrite=True,
        version="v1",
    )
    with open(tmp_filepath, "r+b") as g:
        content = g.read()
        assert content == struct.pack("i" * 4, new_number_of_images, 1, 200, 200)


def test_convert_pink_file_header_v1_v2(
    tmp_path,
    pink_bin_header,
    mosaic_id,
    test_mosaic_dir,
    catalog_filepath,
    number_of_images=40,
    image_height=20,
    image_width=20,
):
    tmp_filepath = tmp_path / pink_bin_header
    hdu = hfits.load_mosaic(mosaic_id, test_mosaic_dir)
    catalog = hfits.read_shimwell_catalog(catalog_filepath, reduced=True)
    catalog_subset = catalog[catalog.Mosaic_ID.str.contains(mosaic_id, regex=False)][
        :number_of_images
    ]
    list_of_coordinates = list(
        map(
            fits.WCSCoordinates,
            np.array(catalog_subset[["RA", "DEC"]].values.tolist())[:, 0],
            np.array(catalog_subset[["RA", "DEC"]].values.tolist())[:, 1],
        )
    )
    hfits.write_mosaic_objects_to_pink_file_v2(
        tmp_filepath, hdu=hdu, coordinates=list_of_coordinates, image_size=20
    )

    pink.convert_pink_file_header_v1_v2(tmp_filepath, v1_to_v2=False)
    with open(tmp_filepath, "rb") as f:
        content = f.read(16)
        assert content == struct.pack(
            "i" * 4, number_of_images, 1, image_width, image_height
        )

    pink.convert_pink_file_header_v1_v2(tmp_filepath, v1_to_v2=True)
    with open(tmp_filepath, "rb") as f:
        content = f.read(32)
        assert content == struct.pack(
            "i" * 8, 2, 0, 0, number_of_images, 0, 2, image_height, image_width
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


def test_write_mosaic_objects_to_pink_file_v2_with_image_containing_nan(
    tmp_path,
    mosaic_hdu_and_wcs_with_nan,
    example_object_world_coordinates,
):

    hdu, wcs = mosaic_hdu_and_wcs_with_nan
    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    number_of_written_images = hfits.write_mosaic_objects_to_pink_file_v2(
        filepath=tmp_filepath,
        hdu=hdu,
        coordinates=[example_object_world_coordinates],
        image_size=20,
    )

    assert number_of_written_images == 0
    assert os.path.exists(tmp_filepath)


def test_write_mosaic_objects_to_pink_file_v2_with_image_containing_nan_fill_nans(
    tmp_path,
    mosaic_hdu_and_wcs_with_nan,
    example_object_world_coordinates,
):

    hdu, _ = mosaic_hdu_and_wcs_with_nan
    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    number_of_written_images = hfits.write_mosaic_objects_to_pink_file_v2(
        filepath=tmp_filepath,
        hdu=hdu,
        coordinates=[example_object_world_coordinates],
        image_size=20,
        fill_nan=True,
    )

    assert number_of_written_images == 1
    assert os.path.exists(tmp_filepath)


def test_write_catalog_to_pink_file_full_image_95px(
    tmp_path, test_mosaic_dir, catalog_p205_p218_full_95px
):

    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    number_of_images = hfits.write_catalog_objects_pink_file_v2(
        filepath=tmp_filepath,
        catalog=catalog_p205_p218_full_95px,
        mosaic_path=test_mosaic_dir,
        image_size=95,
    )

    assert tmp_filepath.exists()
    assert number_of_images == 5


def test_write_catalog_to_pink_file_with_partial_images_95px(
    tmp_path, test_mosaic_dir, catalog_p205_p218_95px
):

    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    number_of_images = hfits.write_catalog_objects_pink_file_v2(
        filepath=tmp_filepath,
        catalog=catalog_p205_p218_95px,
        mosaic_path=test_mosaic_dir,
        image_size=95,
    )

    assert tmp_filepath.exists()
    assert number_of_images == 5
