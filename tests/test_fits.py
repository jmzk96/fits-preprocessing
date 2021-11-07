import pytest

from hda_fits import fits,pink
from hda_fits.logging_config import logging
import struct

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def test_read_shimwell_catalog(catalog_filepath):
    catalog = fits.read_shimwell_catalog(catalog_filepath)
    assert catalog is not None


def test_load_mosaic(mosaic_id, test_data_dir):
    catalog = fits.load_mosaic(mosaic_id=mosaic_id, path=test_data_dir)
    assert catalog is not None


def test_square_cutout_creation(mosaic_hdu_and_wcs, example_object_world_coordinates):
    hdu, wcs = mosaic_hdu_and_wcs
    size = 200
    cutout = fits.create_cutout2D(
        hdu=hdu, coordinates=example_object_world_coordinates, size=size
    )

    log.info(cutout.data.shape)
    assert cutout.data.shape == (size, size)


def test_rectangular_cutout_creation(
    mosaic_hdu_and_wcs, example_object_world_coordinates
):
    hdu, wcs = mosaic_hdu_and_wcs
    size = fits.RectangleSize(image_height=100, image_width=200)
    cutout = fits.create_cutout2D(
        hdu=hdu, coordinates=example_object_world_coordinates, size=size
    )

    assert cutout.data.shape == size

def test_write_pink_file_v2_header(
    test_pink_header,
    number_of_images = 3,
    image_height = 200 ,
    image_width = 200,
    overwrite = True
):
    pink.write_pink_file_v2_header(test_pink_header,
    number_of_images,
    image_height,
    image_width,
    overwrite)
    with open(test_pink_header,"r+b") as g:
        content = g.read()
        assert content == struct.pack( "i" * 8, 2, 0, 0, 3, 0, 2, 200, 200)


def test_write_pink_file_v2_data(
    test_pink_data,
    mosaic_hdu_and_wcs,
    example_object_world_coordinates
):  
    hdu, wcs = mosaic_hdu_and_wcs
    example_array = fits.create_cutout2D_as_flattened_numpy_array(hdu,
    example_object_world_coordinates,
    20,wcs)
    pink.write_pink_file_v2_data(test_pink_data,example_array)
    with open(test_pink_data,"r+b") as g:
        content = g.read()
        assert content == struct.pack("%sf" % example_array.size, *example_array.tolist())

