from pathlib import Path

import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from hda_fits.fits import WCSCoordinates


@pytest.fixture(scope="module")
def catalog_filename():
    return "catalog_shimwell_reduced.srl.fits"


@pytest.fixture(scope="module")
def mosaic_id():
    return "P205+55_test_cutout"


@pytest.fixture(scope="module")
def mosaic_filename(mosaic_id):
    return mosaic_id + "-mosaic.fits"

@pytest.fixture(scope="module")
def pink_bin_header():
    return "test_pink_header.bin"

@pytest.fixture(scope="module")
def pink_bin_data():
    return "test_pink_data.bin"

@pytest.fixture(scope="module")
def test_data_dir():
    path = Path(__file__).parent / "data"
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def catalog_filepath(test_data_dir, catalog_filename):
    path = test_data_dir / catalog_filename
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def mosaic_filepath(test_data_dir, mosaic_filename):
    path = test_data_dir / mosaic_filename
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def shimwell_catalog(catalog_filepath):
    table = Table.read(catalog_filepath)
    assert table is not None
    return table

@pytest.fixture(scope="module")
def test_pink_header(test_data_dir,pink_bin_header):
    path = test_data_dir / pink_bin_header
    assert path.exists()
    return path

@pytest.fixture(scope="module")
def test_pink_data(test_data_dir,pink_bin_data):
    path = test_data_dir / pink_bin_data
    assert path.exists()
    return path

@pytest.fixture(scope="module")
def mosaic_hdu_and_wcs(mosaic_filepath):
    hdu = fits.open(mosaic_filepath)[0]
    wcs = WCS(hdu.header)
    assert hdu is not None
    assert wcs is not None
    return hdu, wcs


@pytest.fixture(scope="module")
def example_object_world_coordinates():
    return WCSCoordinates(207.1492755176664, 55.1906127688414)


@pytest.fixture(scope="module")
def example_object_world_coordinates_outside():
    return WCSCoordinates(50, 1)
