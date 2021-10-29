import pytest

from hda_fits import fits
from hda_fits.logging_config import logging

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
    size = fits.RectangleSize(width=100, height=200)
    cutout = fits.create_cutout2D(
        hdu=hdu, coordinates=example_object_world_coordinates, size=size
    )

    assert cutout.data.shape == size
