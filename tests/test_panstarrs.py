import os

from hda_fits import fits as hfits
from hda_fits import panstarrs as ps
from hda_fits.logging_config import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def test_length_get_images_panstarrs(tmp_path, catalog_filepath, missing_filter_df):
    # test length of reduced shimwell catalog
    reduced_shimwell = hfits.read_shimwell_catalog(catalog_filepath)
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
