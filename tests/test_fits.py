from hda_fits import fits
from hda_fits.logging_config import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def test_read_shimwell_catalog(catalog_filepath):
    catalog = fits.read_shimwell_catalog(catalog_filepath)
    assert catalog is not None


def test_load_mosaic(mosaic_id, test_mosaic_dir):
    catalog = fits.load_mosaic(mosaic_id=mosaic_id, path=test_mosaic_dir)
    assert catalog is not None


def test_square_cutout_creation(mosaic_hdu_and_wcs, example_object_world_coordinates):
    hdu, wcs = mosaic_hdu_and_wcs
    size = 200
    cutout = fits.create_cutout2D(
        hdu=hdu, coordinates=example_object_world_coordinates, size=size
    )

    log.info(cutout.data.shape)
    assert cutout.data.shape == (size, size)


def test_catalog_shape_for_objects(mosaic_ids, shimwell_catalog_df):
    catalog = shimwell_catalog_df
    P205_catalog_CM = catalog[
        (catalog.Mosaic_ID.str.contains(mosaic_ids[0], regex=False))
        & (catalog.S_Code.str.contains("C|M", regex=True))
    ]
    P218_catalog_CM = catalog[
        (catalog.Mosaic_ID.str.contains(mosaic_ids[1], regex=False))
        & (catalog.S_Code.str.contains("C|M", regex=True))
    ]
    assert not P205_catalog_CM.empty
    assert not P218_catalog_CM.empty
    assert P205_catalog_CM.shape == (460, 29)
    assert P218_catalog_CM.shape == (499, 29)


def test_get_correct_coordinates_and_sizes(
    mosaic_id,
    test_mosaic_dir,
    shimwell_catalog_df,
    first_and_last_4_RectangleSizes,
    first_and_last_4_WCSCoordinates,
):
    hdu_header = fits.load_mosaic(mosaic_id, test_mosaic_dir)
    catalog = shimwell_catalog_df
    coordinates, sizes = fits.get_sizes_of_object_selection(hdu_header, catalog)
    assert sizes[:4] == first_and_last_4_RectangleSizes[0]
    assert sizes[-4:] == first_and_last_4_RectangleSizes[1]
    assert coordinates[:4] == first_and_last_4_WCSCoordinates[0]
    assert coordinates[-4:] == first_and_last_4_WCSCoordinates[1]


def test_get_correct_catalog_subset(
    mosaic_id, test_mosaic_dir, catalog_filepath, type_list, shimwell_catalog_df
):
    catalog_subset = shimwell_catalog_df.loc[
        (shimwell_catalog_df.Mosaic_ID.str.contains(mosaic_id, regex=False))
        & (shimwell_catalog_df.S_Code.str.contains("C|M", regex=True)),
        :,
    ]
    catalog_test = fits.get_sizes_of_objects(
        mosaic_id, test_mosaic_dir, catalog_filepath, type_list
    )
    assert catalog_test is not None
    assert not catalog_subset.empty
    assert len(catalog_subset) == len(catalog_test[0])
    assert len(catalog_subset) == len(catalog_test[1])


def test_rectangular_cutout_creation(
    mosaic_hdu_and_wcs, example_object_world_coordinates
):
    hdu, wcs = mosaic_hdu_and_wcs
    size = fits.RectangleSize(image_height=100, image_width=200)
    cutout = fits.create_cutout2D(
        hdu=hdu, coordinates=example_object_world_coordinates, size=size
    )

    assert cutout.data.shape == size
