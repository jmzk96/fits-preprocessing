from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import hda_fits as hf
from hda_fits.fits import RectangleSize, WCSCoordinates
from hda_fits.logging_config import logging

log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def catalog_filename():
    return "catalog_shimwell_reduced.srl.fits"


@pytest.fixture(scope="module")
def mosaic_id():
    return "P205+55"


@pytest.fixture(scope="module")
def mosaic_ids():
    return ["P205+55", "P218+55"]


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
def test_mosaic_dir(test_data_dir):
    path = test_data_dir / "reduced_mosaics"
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def catalog_filepath(test_data_dir, catalog_filename):
    path = test_data_dir / catalog_filename
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def mosaic_filepath(test_mosaic_dir, mosaic_filename):
    path = test_mosaic_dir / mosaic_filename
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def shimwell_catalog(catalog_filepath):
    table = Table.read(catalog_filepath)
    assert table is not None
    return table


@pytest.fixture(scope="module")
def shimwell_catalog_df(catalog_filepath, shimwell_catalog):
    catalog = hf.read_shimwell_catalog(catalog_filepath)
    assert catalog is not None
    assert catalog.shape == shimwell_catalog.to_pandas().shape
    return catalog


@pytest.fixture(scope="module")
def mosaic_hdu_and_wcs(mosaic_filepath):
    hdu = fits.open(mosaic_filepath)[0]
    wcs = WCS(hdu.header)
    assert hdu is not None
    assert wcs is not None
    return hdu, wcs


@pytest.fixture(scope="module")
def mosaic_hdu_and_wcs_with_nan(mosaic_hdu_and_wcs, example_object_pixel_coordinates):
    hdu, wcs = mosaic_hdu_and_wcs
    hdu_nan = hdu.copy()
    hdu_nan.data[tuple(example_object_pixel_coordinates)] = np.nan
    assert np.isnan(hdu_nan.data).any()
    return hdu_nan, wcs


@pytest.fixture(scope="module")
def example_object_world_coordinates():
    return WCSCoordinates(205.07668792, 54.91164316)


@pytest.fixture(scope="module")
def example_object_pixel_coordinates(
    example_object_world_coordinates, mosaic_hdu_and_wcs
):
    _, wcs = mosaic_hdu_and_wcs
    coords = wcs.wcs_world2pix([example_object_world_coordinates], 0)
    return coords.astype("int").tolist()


@pytest.fixture(scope="module")
def example_object_world_coordinates_outside():
    return WCSCoordinates(50, 1)


@pytest.fixture(scope="module")
def sources_closest_to_218_center_full_95px():
    return [
        "ILTJ143425.75+545756.4",
        "ILTJ143410.87+545843.2",
        "ILTJ143426.08+545730.5",
    ]


@pytest.fixture(scope="module")
def sources_closest_to_218_center_partial_95px():
    return [
        "ILTJ143416.01+545700.0",
        "ILTJ143428.48+545728.7",
        "ILTJ143419.81+545616.1",
    ]


@pytest.fixture(scope="module")
def type_list():
    return ["C", "M"]


@pytest.fixture(scope="module")
def first_and_last_4_RectangleSizes():
    return [
        [
            RectangleSize(
                image_height=4.7799646097060045, image_width=4.7799646097060045
            ),
            RectangleSize(
                image_height=4.519641190258223, image_width=4.519641190258223
            ),
            RectangleSize(
                image_height=4.891342053003956, image_width=4.891342053003956
            ),
            RectangleSize(
                image_height=7.692137862465569, image_width=7.692137862465569
            ),
        ],
        [
            RectangleSize(
                image_height=71.51529374562816, image_width=71.51529374562816
            ),
            RectangleSize(
                image_height=45.733983926715204, image_width=45.733983926715204
            ),
            RectangleSize(
                image_height=76.97044763491165, image_width=76.97044763491165
            ),
            RectangleSize(
                image_height=56.69196613908916, image_width=56.69196613908916
            ),
        ],
    ]


@pytest.fixture(scope="module")
def first_and_last_4_WCSCoordinates():
    return [
        [
            WCSCoordinates(RA=220.82511725201545, DEC=54.04997453184971),
            WCSCoordinates(RA=220.8376873437023, DEC=54.34656448042709),
            WCSCoordinates(RA=220.83929502582853, DEC=54.39477141758409),
            WCSCoordinates(RA=220.83230210536595, DEC=54.341771796372804),
        ],
        [
            WCSCoordinates(RA=204.49837507932645, DEC=55.01845918082531),
            WCSCoordinates(RA=204.06229187320707, DEC=54.78647469721186),
            WCSCoordinates(RA=203.41760285716038, DEC=54.02772536519059),
            WCSCoordinates(RA=203.14130732613486, DEC=54.32427759709343),
        ],
    ]


@pytest.fixture(scope="module")
def sources_closest_to_205_center_full_95px():
    return ["ILTJ134018.41+545441.9", "ILTJ134004.60+545356.5"]


@pytest.fixture(scope="module")
def sources_closest_to_205_center_partial_95px():
    return ["ILTJ133940.17+545649.9", "ILTJ134040.88+545808.2"]


@pytest.fixture(scope="module")
def sources_p205_p218_full_95px(
    sources_closest_to_205_center_full_95px, sources_closest_to_218_center_full_95px
):
    return (
        sources_closest_to_205_center_full_95px
        + sources_closest_to_218_center_full_95px
    )


@pytest.fixture(scope="module")
def sources_p205_p218_partial_95px(
    sources_closest_to_205_center_partial_95px,
    sources_closest_to_218_center_partial_95px,
):
    return (
        sources_closest_to_205_center_partial_95px
        + sources_closest_to_218_center_partial_95px
    )


@pytest.fixture(scope="module")
def sources_p205_p218(
    sources_p205_p218_full_95px,
    sources_p205_p218_partial_95px,
):
    return sources_p205_p218_full_95px + sources_p205_p218_partial_95px


@pytest.fixture(scope="module")
def catalog_p205_p218_full_95px(sources_p205_p218_full_95px, shimwell_catalog_df):
    catalog_subset = shimwell_catalog_df[
        shimwell_catalog_df.Source_Name.isin(sources_p205_p218_full_95px)
    ]
    assert catalog_subset.shape[0] == 5
    return catalog_subset


@pytest.fixture(scope="module")
def catalog_p205_p218_95px(sources_p205_p218, shimwell_catalog_df):
    catalog_subset = shimwell_catalog_df[
        shimwell_catalog_df.Source_Name.isin(sources_p205_p218)
    ]
    assert catalog_subset.shape[0] == 10
    return catalog_subset
