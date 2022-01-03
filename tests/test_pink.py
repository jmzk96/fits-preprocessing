import os
import struct

import hda_fits as hfits
import hda_fits.panstarrs as ps
from hda_fits import fits, pink
from hda_fits.logging_config import logging
from hda_fits.types import Layout, PinkHeader

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
    tmp_path, catalog_p205_p218_full_95px, test_mosaic_dir
):
    tmp_filepath = tmp_path / "test_file.pink"
    image_height = 95
    image_width = 95
    catalog_written = hfits.write_catalog_objects_pink_file_v2(
        filepath=tmp_filepath,
        catalog=catalog_p205_p218_full_95px,
        mosaic_path=test_mosaic_dir,
        image_size=95,
    )
    number_of_images = catalog_written.shape[0]

    pink.convert_pink_file_header_v2_to_v1(tmp_filepath)
    with open(tmp_filepath, "rb") as f:
        content = f.read(16)
        assert content == struct.pack(
            "i" * 4, number_of_images, 1, image_width, image_height
        )
    with open(tmp_filepath, "rb") as f:
        f.seek(16)
        assert len(f.read()) == 180500

    pink.convert_pink_file_header_v1_to_v2(tmp_filepath)
    with open(tmp_filepath, "rb") as f:
        content = f.read(32)
        assert content == struct.pack(
            "i" * 8, 2, 0, 0, number_of_images, 0, 2, image_height, image_width
        )
    with open(tmp_filepath, "rb") as f:
        f.seek(32)
        assert len(f.read()) == 180500


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

    image_was_written = hfits.write_mosaic_objects_to_pink_file_v2(
        filepath=tmp_filepath,
        hdu=hdu,
        coordinates=[c1, c1, c1],
        image_size=20,
    )

    assert sum(image_was_written) == 3
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

    image_was_written = hfits.write_mosaic_objects_to_pink_file_v2(
        filepath=tmp_filepath,
        hdu=hdu,
        coordinates=[c1, c1, c1],
        image_size=fits.RectangleSize(image_height=20, image_width=10),
    )

    assert sum(image_was_written) == 3
    assert os.path.exists(tmp_filepath)


def test_write_mosaic_objects_to_pink_file_v2_with_coordinate_outside_of_mosaic(
    tmp_path,
    mosaic_hdu_and_wcs,
    example_object_world_coordinates_outside,
):

    hdu, wcs = mosaic_hdu_and_wcs
    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    image_was_written = hfits.write_mosaic_objects_to_pink_file_v2(
        filepath=tmp_filepath,
        hdu=hdu,
        coordinates=[example_object_world_coordinates_outside],
        image_size=20,
    )

    assert sum(image_was_written) == 0
    assert os.path.exists(tmp_filepath)


def test_write_mosaic_objects_to_pink_file_v2_with_image_containing_nan(
    tmp_path,
    mosaic_hdu_and_wcs_with_nan,
    example_object_world_coordinates,
):

    hdu, wcs = mosaic_hdu_and_wcs_with_nan
    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    image_was_written = hfits.write_mosaic_objects_to_pink_file_v2(
        filepath=tmp_filepath,
        hdu=hdu,
        coordinates=[example_object_world_coordinates],
        image_size=20,
    )

    assert sum(image_was_written) == 0
    assert os.path.exists(tmp_filepath)


def test_write_mosaic_objects_to_pink_file_v2_with_image_containing_nan_fill_nans(
    tmp_path,
    mosaic_hdu_and_wcs_with_nan,
    example_object_world_coordinates,
):

    hdu, _ = mosaic_hdu_and_wcs_with_nan
    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    image_was_written = hfits.write_mosaic_objects_to_pink_file_v2(
        filepath=tmp_filepath,
        hdu=hdu,
        coordinates=[example_object_world_coordinates],
        image_size=20,
        fill_nan=True,
    )

    assert sum(image_was_written) == 1
    assert os.path.exists(tmp_filepath)


def test_write_catalog_to_pink_file_full_image_95px(
    tmp_path, test_mosaic_dir, catalog_p205_p218_full_95px
):

    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    catalog_written = hfits.write_catalog_objects_pink_file_v2(
        filepath=tmp_filepath,
        catalog=catalog_p205_p218_full_95px,
        mosaic_path=test_mosaic_dir,
        image_size=95,
    )
    number_of_images = catalog_written.shape[0]

    assert tmp_filepath.exists()
    assert number_of_images == 5


def test_write_catalog_to_pink_file_with_partial_images_95px(
    tmp_path, test_mosaic_dir, catalog_p205_p218_95px, sources_p205_p218_full_95px
):

    tmp_filepath = tmp_path / "test_file.pink"

    assert not tmp_filepath.exists()

    catalog_written = hfits.write_catalog_objects_pink_file_v2(
        filepath=tmp_filepath,
        catalog=catalog_p205_p218_95px,
        mosaic_path=test_mosaic_dir,
        image_size=95,
    )
    number_of_images = catalog_written.shape[0]

    assert tmp_filepath.exists()
    assert number_of_images == 5

    sources_found = catalog_written["Source_Name"].values.tolist()
    assert sorted(sources_found) == sorted(sources_p205_p218_full_95px)


def test_read_pink_file_header(test_pink_file):
    header = pink.read_pink_file_header(test_pink_file)

    assert type(header) == PinkHeader
    assert header.version == 2
    assert header.file_type == 0
    assert header.data_type == 0
    assert header.dimensionality == 2
    assert header.number_of_images == 20
    assert header.data_layout == 0
    assert header.layout == Layout(depth=1, height=95, width=95)


def test_read_pink_file_image_first_image(test_pink_file):
    image = pink.read_pink_file_image(test_pink_file, 0)
    assert image.shape == (95, 95)


def test_read_pink_file_image_last_image(test_pink_file):
    image = pink.read_pink_file_image(test_pink_file, 19)
    assert image.shape == (95, 95)


def test_read_pink_file_image_invalid_index(test_pink_file):
    image = None
    try:
        image = pink.read_pink_file_image(test_pink_file, 20)
    except struct.error as e:
        assert e is not None

    assert image is None


def test_write_read_panstarrs_objects_to_pink_file(tmp_path, catalog_filepath):
    panstarrs_catalog = hfits.read_shimwell_catalog(catalog_filepath)
    panstarrs_catalog = panstarrs_catalog[panstarrs_catalog.S_Code.str.contains("C|M")]
    filepath = tmp_path / "test_panstarrs.pink"
    # test function when download is False and fits files already in temp folder
    ps.get_images_panstarrs(catalog=panstarrs_catalog[:3], file_directory=tmp_path)
    pink.write_panstarrs_objects_to_pink_file(
        panstarrs_catalog=panstarrs_catalog[:3],
        filepath=filepath,
        panstarrs_data_path=tmp_path,
        image_size=95,
    )
    header = pink.read_pink_file_header(filepath)
    example_data = pink.read_pink_file_image(filepath, 0)
    assert header.version == 2
    assert header.number_of_images == 2
    assert header.layout == Layout(95, 95, 1)
    assert example_data.shape == (95, 95)
    # test function when download is True and fits files dont exist
    for f in os.listdir(tmp_path):
        os.remove(os.path.join(tmp_path, f))
    pink.write_panstarrs_objects_to_pink_file(
        panstarrs_catalog=panstarrs_catalog[:3],
        filepath=filepath,
        panstarrs_data_path=tmp_path,
        image_size=95,
        download=True,
    )
    header = pink.read_pink_file_header(filepath)
    example_data = pink.read_pink_file_image(filepath, 0)
    assert header.version == 2
    assert header.number_of_images == 2
    assert header.layout == Layout(95, 95, 1)
    assert example_data.shape == (95, 95)
