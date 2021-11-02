import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import astropy.units as u
import os
import numpy as np
import sys


def read_fits(fits_path:str):
    """
    reads files in FITS-format and 
    """
    try:
        fits_file= fits.open(fits_path)
        return fits_file
    except OSError:
        print("Could not read/open file",fits_path)
        sys.exit()
        
def get_header_and_data(hdulist:fits.hdu.hdulist.HDUList):
    print(f"Data from fits header : \n {hdulist[0].header}")
    print(f"Numpy Array of image : \n {hdulist[0].data}")
    
def get_ra_dec_pixels(hdulist):
    ra = hdulist[0].header[12]
    dec = hdulist[0].header[13]
    pixel_ra = hdulist[0].header[10]
    pixel_dec = hdulist[0].header[11]
    return (ra,pixel_ra,dec,pixel_dec)

def plot_whole_sky_image(hdulist):
    wcs = WCS(hdulist[0].header)
    ax = plt.subplot(projection=wcs)
    ax.imshow(hdulist[0].data,vmin=0,vmax=0.0004,origin="lower")

    ax.grid(color="white",ls="solid")
    plt.show()
    
def plot_partial_sky_image(hdulist,coord_ra,coord_dec):
    wcs = WCS(hdulist[0].header)
    coord = SkyCoord(ra=coord_ra*u.degree,dec=coord_dec*u.degree,frame="fk5")
    xc,yc = wcs.world_to_pixel(coord)
    dx=200
    ax = plt.subplot(projection=wcs)
    ax.imshow(hdulist[0].data,vmin=0,vmax=0.0004,origin="lower")
    ax.grid(color="white",ls="solid")
    plt.show()
    
def get_header_and_data_catalogue(hdulist):
    print(f"Data from fits catalogue header : \n {hdulist[1].header}")
    print(f"Numpy Array on data of interesting objects : \n {hdulist[1].data}")

def get_source_names(hdulist):
    return hdulist[1].data.__getitem__("Source_Name")
    
def catalogue_checker(hdulist, source_name:str):
    return source_name in hdulist[1].data.__get__item("Source_Name")

def catalogue_locator(hdulist,source_name:str):
    if catalogue_checker(hdulist,source_name):
        return np.flatnonzero(np.core.defchararry.find(hdulist[1]\
        .data.__getitem__("Source_Name"),source_name)!=-1)[0]
    else:
        return None
    
def get_catalogue_ra_dec(hdulist,source_name):
    try:
        ra = hdulist[1].data[catalogue_locator(hdulist,source_name)].__getitem__('RA      ')
        dec = hdulist[1].data[catalogue_locator(hdulist,source_name)].__getitem__('DEC     ')
        return (ra,dec)
    except TypeError:
        print("Value Type not supported by function")

def scale_checker(fits_image,catalogue,source_name):
    try:
        ra,dec = get_catalogue_ra_dec(catalogue,source_name)
        fits_ra_pxl_max = fits_image[0].header[3]
        fits_dec_pxl_max = fits_image[0].header[4]
        wcs = WCS(fits_image[0].header)
        coord = SkyCoord(ra=ra*u.degree,dec=dec*u.degree,frame="fk5")
        xc,yc = wcs.world_to_pixel(coord)
        if 0.0 <= float(xc) <= fits_ra_pxl_max and 0.0 <= float(yc) <=fits_dec_pxl_max:
            return True
        else:
            return False
    except TypeError:
        print("Value Type not supported by function")
        