import os
import tempfile
import time
from io import StringIO
from time import sleep

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.table import Table

from .logging_config import logging
from .types import WCSCoordinates

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
fitscut = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"


def get_images_panstarrs(
    catalog: pd.DataFrame,
    file_directory: str,
    size: int = 1000,
    filters: str = "gri",
    format: str = "fits",
    imagetypes: str = "stack",
    return_table_only: bool = False,
    **kwargs,
):

    """Query ps1filenames.py service for multiple positions to get a list of images
    This adds a url column to the table to retrieve the cutout.

    tra, tdec = list of positions in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    format = data format (options are "fits", "jpg", or "png")
    imagetypes = list of any of the acceptable image types.  Default is stack;
        other common choices include warp (single-epoch images), stack.wt (weight image),
        stack.mask, stack.exp (exposure time), stack.num (number of exposures),
        warp.wt, and warp.mask.  This parameter can be a list of strings or a
        comma-separated string.

    Returns an astropy table with the results
    """

    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    # if imagetypes is a list, convert to a comma-separated string
    if not isinstance(imagetypes, str):
        imagetypes = ",".join(imagetypes)
    # put the positions in an in-memory file object
    tra = catalog["RA"].tolist()
    tdec = catalog["DEC"].tolist()
    tsource = catalog["Source_Name"].tolist()
    n_bands = len(filters)

    cbuf = StringIO()
    cbuf.write("\n".join(["{} {}".format(ra, dec) for (ra, dec) in zip(tra, tdec)]))
    cbuf.seek(0)
    # use requests.post to pass in positions as a file
    r = requests.post(
        ps1filename, data=dict(filters=filters, type=imagetypes), files=dict(file=cbuf)
    )
    r.raise_for_status()
    tab = Table.read(r.text, format="ascii")

    urlbase = "{}?size={}&format={}".format(fitscut, size, format)
    tab["url"] = [
        "{}&ra={}&dec={}&red={}".format(urlbase, ra, dec, filename)
        for (filename, ra, dec) in zip(tab["filename"], tab["ra"], tab["dec"])
    ]
    pandas_table = Table.to_pandas(tab)
    pandas_table.rename(columns={"ra": "RA", "dec": "DEC"}, inplace=True)
    grouped = pandas_table.groupby(["RA", "DEC"], sort=False).agg(list)
    grouped.reset_index(inplace=True)
    grouped["number_list"] = grouped.apply(lambda x: len(x["filter"]), axis=1)
    grouped["Source_Name"] = tsource
    if not grouped[grouped["number_list"] != n_bands].empty:
        log.warning(
            "Following objects with their coordinates have missing filters (RA,DEC) {}".format(
                grouped[grouped["number_list"] != n_bands].index.tolist()
            )
        )
        log.warning(
            "Source Names with missing filters are: {}".format(
                grouped[grouped["number_list"] != n_bands].Source_Name.tolist()
            )
        )
        grouped_copy = grouped[grouped["number_list"] == n_bands].copy()
        if grouped_copy.empty:
            log.warning("Source doesnt have all {} bands ".format(n_bands))
            return None
        exploded = grouped_copy.explode(["url", "filter"])
        if return_table_only:
            return Table.from_pandas(exploded)
        elif not return_table_only and file_directory:
            return panstarrs_image_loader(
                Table.from_pandas(exploded), file_directory, **kwargs
            )
    exploded = grouped.explode(["url", "filter"])

    if return_table_only:
        return Table.from_pandas(exploded)
    elif not return_table_only and file_directory:
        return panstarrs_image_loader(
            Table.from_pandas(exploded), file_directory, **kwargs
        )


def panstarrs_image_loader(
    astropy_table: Table,
    file_directory: str,
    seperate_channels: bool = True,
    sleep_time: int = 60,
):
    """
    This function takes in an Astropy Table and saves the files from the Pan STARRS API as
    FITS files in a selected file directory.

    If filepath to FITS file exists, the file is skipped and not downloaded.
    Seperate Channels eg. grizy can be downloaded or a combination of theses channels
    can be calculated and downloaded as a single image.
    The sleep time for the requests.get function can also be manually set.

    Returns a saved FITS files saved in specified file_directory and with format:
    {Source_Name}_filter={filter}.fits for seperate channels and {Source_Name}.fits when all channels are combined
    """
    t0 = time.time()
    number_of_missing_images = 0
    number_of_loading_images = 0
    if seperate_channels:
        for row in astropy_table:
            coordinates = WCSCoordinates(row["RA"], row["DEC"])
            filter = row["filter"]
            url = row["url"]
            source_name = row["Source_Name"]
            filename = "{}_filter={}.fits".format(source_name, filter)
            filepath = os.path.join(file_directory, filename)
            if not os.path.exists(filepath):
                try:
                    number_of_loading_images += 1
                    if number_of_loading_images <= 500:
                        r = requests.get(url)
                        if r.status_code == 200:
                            with open(filepath, "wb") as fitsfile:
                                fitsfile.write(r.content)
                        else:
                            number_of_missing_images += 1
                            log.warning(
                                f"Image at coordinates RA:{coordinates.RA} , DEC:{coordinates.DEC} with name:{source_name} at Band:{filter}\
                                    not added to panSTARRS file stream"
                            )
                    else:
                        number_of_loading_images = 0
                        sleep(sleep_time)
                        r = requests.get(url)
                        if r.status_code == 200:
                            with open(filepath, "wb") as fitsfile:
                                fitsfile.write(r.content)
                        else:
                            number_of_missing_images += 1
                            log.warning(
                                f"Image at coordinates RA:{coordinates.RA} , DEC:{coordinates.DEC} with name:{source_name} at Band:{filter}\
                                    not added to panSTARRS file stream"
                            )
                except requests.exceptions.ConnectionError:
                    r.status_code = "Connection refused"
                    log.warning(
                        f"Image at coordinates RA:{coordinates.RA} , DEC:{coordinates.DEC} with name:{source_name} at Band:{filter}\
                                    not added to panSTARRS file stream"
                    )
        log.info(
            "{:.1f} s: loaded {} FITS files from {} expected FITS files at different bands".format(
                time.time() - t0,
                len(astropy_table) - number_of_missing_images,
                len(astropy_table),
            )
        )

    else:
        aggregated_table_by_coords = astropy_table.group_by(
            ["RA", "DEC"]
        ).groups.aggregate(list)
        for row in aggregated_table_by_coords:
            source_name = row["Source_Name"][0]
            coordinates = WCSCoordinates(row["RA"], row["DEC"])
            filename = "{}.fits".format(source_name)
            filepath = os.path.join(file_directory, filename)

            list_of_channel_data = []
            if not os.path.exists(filepath):
                for filter, url in zip(row["filter"], row["url"]):
                    try:
                        number_of_loading_images += 1
                        if number_of_loading_images <= 500:
                            r = requests.get(url)
                            if r.status_code == 200:

                                tf = tempfile.TemporaryFile()
                                tf.write(r.content)
                                tf.seek(0)
                                header = fits.open(tf, mode="update")[0].header
                                data = fits.open(tf, mode="update")[0].data
                                list_of_channel_data.append(data)
                                tf.close()
                            else:
                                number_of_missing_images += 1
                                log.warning(
                                    f"Image at coordinates RA:{coordinates.RA} , DEC:{coordinates.DEC} with name:{source_name} at Band:{filter}\
                                        not added to panSTARRS file stream"
                                )
                        else:
                            number_of_loading_images = 0
                            sleep(sleep_time)
                            r = requests.get(url)
                            if r.status_code == 200:

                                tf = tempfile.TemporaryFile()
                                tf.write(r.content)
                                tf.seek(0)
                                header = fits.open(tf, mode="update")[0].header
                                data = fits.open(tf, mode="update")[0].data
                                list_of_channel_data.append(data)
                                tf.close()
                            else:
                                number_of_missing_images += 1
                                log.warning(
                                    f"Image at coordinates RA:{coordinates.RA} , DEC:{coordinates.DEC} with name:{source_name} at Band:{filter}\
                                        not added to panSTARRS file stream"
                                )

                    except requests.exceptions.ConnectionError:
                        r.status_code = "Connection refused"
                        number_of_missing_images += 1
                        log.warning(
                            f"Image at coordinates RA:{coordinates.RA} , DEC:{coordinates.DEC} with name:{source_name} at Band:{filter}\
                            not added to panSTARRS file stream"
                        )

                image_info = np.mean(np.array(list_of_channel_data), axis=0)
                fits.writeto(filepath, image_info, header, overwrite=True)

        log.info(
            "{:.1f} s: loaded  {} FITS files from {} expected FITS files at different bands".format(
                time.time() - t0,
                len(aggregated_table_by_coords) - number_of_missing_images,
                len(aggregated_table_by_coords),
            )
        )


def load_panstarrs_file(
    catalog: pd.DataFrame, source_name: str, path: str, download: bool = False
):
    """
    Load Pan-STARRS files in specified path, if download is set to True, the function will try to download the FITS files
    through the Pan-STARRS API. A valid source_name is required such as  ILTJ122003.20+490932.9 from the LoTTs catalog

    this function only loads fits files of single channels and not whole images.
    """
    primary_hdus = []
    bands = ["r", "g", "i"]
    for band in bands:
        filepath = os.path.join(path, f"{source_name}_filter={band}.fits")
        if not os.path.exists(filepath) and download:
            catalog_download = catalog[catalog.Source_Name == source_name]
            get_images_panstarrs(catalog_download, path)
            try:
                primary_hdu = fits.open(filepath)[0]
                primary_hdus.append(primary_hdu)
            except FileNotFoundError as e:
                log.debug(e)
                return None
        try:
            log.debug(f"Loading {filepath}")
            primary_hdu = fits.open(filepath)[0]
            primary_hdus.append(primary_hdu)
        except FileNotFoundError as e:
            log.warning(e)
            return None
    return primary_hdus
