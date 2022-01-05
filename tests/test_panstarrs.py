import os

from astropy.io.fits.hdu.image import PrimaryHDU

from hda_fits import fits as hfits
from hda_fits import panstarrs as ps
from hda_fits import pink
from hda_fits.logging_config import logging
from hda_fits.types import Layout

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def test_length_get_images_panstarrs(tmp_path, catalog_filepath, missing_filter_df):
    # test length of reduced shimwell catalog
    reduced_shimwell = hfits.read_shimwell_catalog(catalog_filepath)
    reduced_shimwell = reduced_shimwell[reduced_shimwell.S_Code.str.contains("C|M")]
    assert (
        len(ps.get_images_panstarrs(reduced_shimwell, tmp_path, return_table_only=True))
        == 2877
    )
    assert (
        len(
            ps.get_images_panstarrs(
                reduced_shimwell, tmp_path, return_table_only=True, filters="griy"
            )
        )
        == 3836
    )
    # test existence of dataframe with missing filters
    assert (
        ps.get_images_panstarrs(missing_filter_df, tmp_path, return_table_only=True)
        is None
    )


def test_panstarrs_image_loader(tmp_path, catalog_filepath):
    reduced_shimwell = hfits.read_shimwell_catalog(catalog_filepath)
    ps.get_images_panstarrs(reduced_shimwell.iloc[:3], tmp_path)
    assert (
        len(
            [
                name
                for name in os.listdir(tmp_path)
                if os.path.isfile(os.path.join(tmp_path, name))
            ]
        )
        == 9
    )
    ps.get_images_panstarrs(reduced_shimwell.iloc[:4], tmp_path)
    assert (
        len(
            [
                name
                for name in os.listdir(tmp_path)
                if os.path.isfile(os.path.join(tmp_path, name))
            ]
        )
        == 12
    )
    for f in os.listdir(tmp_path):
        os.remove(os.path.join(tmp_path, f))
    ps.get_images_panstarrs(
        reduced_shimwell.iloc[:3], tmp_path, seperate_channels=False
    )
    assert (
        len(
            [
                name
                for name in os.listdir(tmp_path)
                if os.path.isfile(os.path.join(tmp_path, name))
            ]
        )
        == 3
    )


def test_panstarrs_loader(tmp_path, catalog_filepath):
    reduced_shimwell = hfits.read_shimwell_catalog(catalog_filepath)
    example_source_name = reduced_shimwell.iloc[:1].Source_Name.values[0]
    primary_hdus = ps.load_panstarrs_file(
        reduced_shimwell, example_source_name, tmp_path, download=True
    )
    assert primary_hdus is not None
    assert len(primary_hdus) == 3
    for hdu in primary_hdus:
        assert isinstance(hdu, PrimaryHDU)


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
