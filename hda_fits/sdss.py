import urllib
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.io.fits.hdu.image import PrimaryHDU
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from reproject import reproject_interp

from hda_fits import fits as hfits

from .logging_config import logging
from .types import RectangleSize, SDSSFields, WCSCoordinates

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


CROSSMATCH_CATALOG_FILENAME = "shimwell_sdss_crossmatch_catalog.parquet"
SDSS_FIELD_COLUMNS = ["run", "cam_col", "field"]
MOSAIC_CDELT1 = -0.00041666666666666
MOSAIC_CDELT2 = 0.000416666666666666


def query_sdss_fields(
    coordinates: WCSCoordinates, size: float = 1, number_of_entries: int = 1
) -> Tuple[pd.DataFrame, SDSSFields]:  # all in degree

    fmt = "csv"
    default_url = "http://skyserver.sdss3.org/public/en/tools/search/x_sql.aspx"
    ra_max = coordinates.RA + size
    ra_min = coordinates.RA - size
    dec_max = coordinates.DEC + size
    dec_min = coordinates.DEC - size

    query = f"""
        SELECT TOP {number_of_entries} fieldID as field_id, run, camCol as cam_col, field, ra, dec, rerun,
        ra - {coordinates.RA} as delta_ra,
        dec - {coordinates.DEC} as delta_dec,
            (ra - {coordinates.RA}) * (ra - {coordinates.RA}) +
            (dec - {coordinates.DEC}) * (dec - {coordinates.DEC})
        as squared_distance
        FROM Field
        WHERE ra  BETWEEN {ra_min}  AND {ra_max}
        AND   dec BETWEEN {dec_min} AND {dec_max}
        ORDER BY (ra - {coordinates.RA}) * (ra - {coordinates.RA}) + (dec - {coordinates.DEC}) * (dec - {coordinates.DEC})
        """

    params = urllib.parse.urlencode({"cmd": query, "format": fmt})
    url_opened = urllib.request.urlopen(default_url + "?%s" % params)
    lines = url_opened.readlines()
    df = bytes_to_pandas(lines)
    fields = SDSSFields(*df[["run", "cam_col", "field"]].values[0].tolist())
    log.info(f"Scraped fields {fields}")
    return df, fields


def bytes_to_pandas(list_of_bytes: List) -> pd.DataFrame:
    """
    This function turns the ByteStrings (intially from the function getSDSSfields)
    to a pandas Dataframe
    """
    joined_byte_list = b"".join(list_of_bytes)
    df = pd.read_csv(BytesIO(joined_byte_list))
    df.reset_index(inplace=True)
    headers = df.iloc[0]
    df_with_proper_headers = pd.DataFrame(df.values[1:], columns=headers)
    df_with_proper_headers = df_with_proper_headers.astype(
        {
            "field_id": str,
            "run": int,
            "cam_col": int,
            "field": int,
            "ra": float,
            "dec": float,
            "rerun": int,
            "delta_ra": float,
            "delta_dec": float,
            "squared_distance": float,
        }
    )
    return df_with_proper_headers


def download_sdss_file(fields: SDSSFields, band: str, filepath: Path):
    """Download SDSS file"""
    run, cam_col, field = fields
    filename = f"frame-{band}-{run:06d}-{cam_col}-{field:04d}.fits.bz2"
    http = f"https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/{run}/{cam_col}/{filename}"

    with open(filepath, "wb") as f:
        content = requests.get(http).content
        f.write(content)


def load_sdss_field_files(fields: SDSSFields, path: str, download: bool = False):
    primary_hdus = []
    bands = ["r", "g", "i"]

    _path = Path(path)
    dirpath = _path / "raw" / str(fields)
    dirpath.mkdir(parents=True, exist_ok=True)

    for band in bands:
        filepath = dirpath / f"{fields}-{band}.fits"
        if not filepath.exists() and download:
            download_sdss_file(fields=fields, band=band, filepath=filepath)
        else:
            log.debug(f"File {filepath} exists. Not downloading..")

        hdu = fits.open(filepath)[0]
        primary_hdus.append(hdu)

    return primary_hdus


def download_sdss_field_files(fields: SDSSFields, path: str):
    bands = ["r", "g", "i"]
    _path = Path(path)
    dirpath = _path / "raw" / str(fields)
    dirpath.mkdir(parents=True, exist_ok=True)

    for band in bands:
        filepath = dirpath / f"{fields}-{band}.fits"
        if not filepath.exists():
            log.debug(f"Downloading {fields} to {filepath}")
            download_sdss_file(fields=fields, band=band, filepath=filepath)
        else:
            log.debug(f"File {filepath} exists. Not downloading..")


def download_field_files_in_crossmatch_catalog(
    crossmatch_catalog: pd.DataFrame, path: str, delay_seconds: float = None
):
    fields = extract_sdss_fields_from_catalog(crossmatch_catalog)
    fields_unique = list(set(fields))

    total_number_of_fields = len(fields_unique)

    for i, field in enumerate(fields_unique):
        # tmp_path = tempfile.TemporaryPath()
        # primary_hdus = load_sdss_field_files(field, tmp_path, download=True)
        # - create merge
        # - reproject
        # - save merged image
        log.info(f"[{i}/{total_number_of_fields}] Downloading field {field}")
        download_sdss_field_files(field, path)
        if delay_seconds:
            sleep(delay_seconds)


def create_sdss_image_array(r_band: PrimaryHDU, g_band: PrimaryHDU, b_band: PrimaryHDU):
    r = r_band.data
    g = g_band.data
    b = b_band.data
    optical_array = r / 3 + g / 3 + b / 3
    return optical_array


def create_lupton_rgb_image(rgb_images: List[np.ndarray], merge: bool = True):
    rgb_lupton = make_lupton_rgb(*rgb_images)
    if merge:
        rgb_lupton = rgb_lupton.mean(axis=2)
    return rgb_lupton


def create_rgb_image(rgb_images: List[np.ndarray], merge: bool = True):
    rgb_image = np.dstack(rgb_images)
    if merge:
        rgb_image = rgb_image.mean(axis=2)
    return rgb_image


def create_reprojected_rgb_image(
    primary_hdus: List[PrimaryHDU],
    coordinates: WCSCoordinates,
    image_size: Union[int, RectangleSize],
    merge: bool = True,
    use_lupton_algorithm: bool = False,
) -> np.ndarray:
    projected_arrays = []
    for hdu in primary_hdus:
        projected_array = reproject_image_array(hdu, coordinates, image_size)
        projected_arrays.append(projected_array)

    if use_lupton_algorithm:
        rgb_image = create_lupton_rgb_image(projected_arrays, merge=merge)
    else:
        rgb_image = create_rgb_image(projected_arrays, merge)

    return rgb_image


def create_sdss_fits_file(data: np.ndarray, filepath: str):
    hdu = PrimaryHDU(data)
    hdu = HDUList([hdu])
    hdu.writeto(filepath)


def reproject_image_array(
    hdu: PrimaryHDU,
    coordinates: WCSCoordinates,
    image_size: Union[int, RectangleSize],
    resolution_delt1: float = MOSAIC_CDELT1,
    resolution_delt2: float = MOSAIC_CDELT2,
) -> np.ndarray:
    """
    This function gives back a reprojection or rather a rotated array of the input image array
    based on the SIN/Orthographic projection in the World Coordinate System
    """
    if isinstance(image_size, int):
        image_size = RectangleSize(image_size, image_size)
    wcs_input_dict = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CDELT1": resolution_delt1,
        "CDELT2": resolution_delt2,
        "CRVAL1": coordinates.RA,
        "CRVAL2": coordinates.DEC,
        "NAXIS1": image_size.image_width,
        "NAXIS2": image_size.image_height,
        "CRPIX1": image_size.image_width / 2.0,
        "CRPIX2": image_size.image_height / 2.0,
    }
    wcs_setup = WCS(wcs_input_dict)
    reprojected_array = reproject_interp(
        input_data=hdu,
        output_projection=wcs_setup,
        shape_out=[image_size.image_width, image_size.image_height],
        return_footprint=False,
    )
    return reprojected_array


def extract_sdss_fields_from_catalog(crossmatch_catalog: pd.DataFrame) -> List:
    return crossmatch_catalog.apply(
        lambda row: SDSSFields(
            run=row["run"], cam_col=row["cam_col"], field=row["field"]
        ),
        axis=1,
    ).values.tolist()


def create_empty_crossmatch_catalog(
    path: str,
    path_shimwell: str,
    overwrite: bool = False,
):

    filepath_crossmatch_catalog = Path(path) / CROSSMATCH_CATALOG_FILENAME
    if filepath_crossmatch_catalog.exists() and not overwrite:
        log.error("Crossmatch crossmatch_catalog already exists. Returning..")
        return

    shimwell_catalog = hfits.read_shimwell_catalog(path_shimwell, reduced=True)

    shimwell_catalog["run"] = None
    shimwell_catalog["cam_col"] = None
    shimwell_catalog["field"] = None

    shimwell_catalog = shimwell_catalog.astype(
        {"run": "Int64", "cam_col": "Int64", "field": "Int64"}
    )

    log.info(
        f"Writing empty crossmatch crossmatch_catalog to {filepath_crossmatch_catalog}"
    )
    shimwell_catalog.to_parquet(filepath_crossmatch_catalog)


def load_crossmatch_catalog(path: str):
    filepath_crossmatch_catalog = Path(path) / CROSSMATCH_CATALOG_FILENAME
    return pd.read_parquet(filepath_crossmatch_catalog)


def fill_sdss_shimwell_crossmatch_catalog(
    path_crossmatch_catalog: str,
    shimwell_subset: pd.DataFrame = None,
    delay_seconds: float = None,
) -> pd.DataFrame:

    crossmatch_catalog = load_crossmatch_catalog(path_crossmatch_catalog)

    if shimwell_subset is None:
        crossmatch_catalog_subset = crossmatch_catalog
    else:
        crossmatch_catalog_subset = crossmatch_catalog[
            crossmatch_catalog["Source_Name"].isin(shimwell_subset["Source_Name"])
        ]

    index_empty_fields = (
        crossmatch_catalog_subset[SDSS_FIELD_COLUMNS].isnull().any(axis=1)
    )
    crossmatch_catalog_unmatched = crossmatch_catalog_subset[index_empty_fields]
    crossmatch_objects = crossmatch_catalog_unmatched.apply(
        lambda row: {
            "Source_Name": row["Source_Name"],
            "wcs_coordinates": WCSCoordinates(row["RA"], row["DEC"]),
        },
        axis=1,
    ).values.tolist()

    for cmo in crossmatch_objects:
        source_name, wcs_coordinates = cmo.values()

        log.info(f"Querying sdss fields for Source: {source_name} @ {wcs_coordinates}")
        _, fields = query_sdss_fields(wcs_coordinates)

        crossmatch_catalog.loc[
            crossmatch_catalog["Source_Name"] == source_name, SDSS_FIELD_COLUMNS
        ] = tuple(fields)

        crossmatch_catalog.to_parquet(
            Path(path_crossmatch_catalog) / CROSSMATCH_CATALOG_FILENAME
        )

        if delay_seconds:
            sleep(delay_seconds)


def extract_crossmatch_attributes(catalog: pd.DataFrame):
    return catalog.apply(
        lambda row: {
            "Source_Name": row["Source_Name"],
            "wcs_coordinates": WCSCoordinates(row["RA"], row["DEC"]),
            "sdss_fields": SDSSFields(
                run=row["run"], cam_col=row["cam_col"], field=row["field"]
            ),
        },
        axis=1,
    ).values.tolist()
