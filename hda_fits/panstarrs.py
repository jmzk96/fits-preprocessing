import os
import tempfile
import time
from io import StringIO

import numpy as np
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
    tra: list,
    tdec: list,
    file_directory: str,
    size: int = 240,
    filters: str = "grizy",
    format: str = "fits",
    imagetypes: str = "stack",
    return_table_only: bool = False,
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
    if return_table_only:
        return tab
    else:
        panstarrs_image_loader(tab, file_directory)


def panstarrs_image_loader(astropy_table: Table, file_directory):
    t0 = time.time()
    number_of_missing_images = 0
    aggregated_table_by_coords = astropy_table.group_by(["ra", "dec"]).groups.aggregate(
        list
    )
    for row in aggregated_table_by_coords:

        coordinates = WCSCoordinates(row["ra"], row["dec"])
        filename = "t{:08.4f}{:+07.4f}.fits".format(coordinates.RA, coordinates.DEC)
        filepath = os.path.join(file_directory, filename)

        list_of_channel_data = []
        for filter, url in zip(row["filter"], row["url"]):
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
                    f"Image at coordinates RA:{coordinates.RA} , DEC:{coordinates.DEC} at Band:{filter}\
                     not added to panSTARRS file stream"
                )
        image_info = np.mean(np.array(list_of_channel_data), axis=0)
        fits.writeto(filepath, image_info, header, overwrite=True)

    log.info(
        "{:.1f} s: retrieved {} FITS files for {} positions".format(
            time.time() - t0,
            len(aggregated_table_by_coords) - number_of_missing_images,
            len(aggregated_table_by_coords),
        )
    )
