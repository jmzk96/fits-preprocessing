#!/opt/local/bin/python3.7

import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import astropy.units as u


# for fits IO see
#  https://docs.astropy.org/en/stable/io/fits/index.html
hdu1 = fits.open("LOFAR_HBA_T1_DR1_catalog_v1.0.srl.fits")

# print some information from the fits table
hdu1.info()
hdr = hdu1[1].header
print(repr(hdr))
data = hdu1[1].data
print(data[0])


# I selected one source and plot a cut with 400 pixel side length, centered on source
#    https://docs.astropy.org/en/stable/visualization/wcsaxes/index.html
# read how pixel coordinates translate to sky coordinates from the fits image
hdu = fits.open("P205+55-mosaic.fits")[0]
wcs = WCS(hdu.header)

# using sky coordinates
#   https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html
c1_string =  "13:40:38.2 +55:50:21"   # coordinates of the source I selected from  P205+55-mosaic.fits
c1 = SkyCoord( c1_string, frame=FK5, unit=(u.hourangle, u.deg) )
print(c1)


# from world coordinats to pixel
#    https://docs.astropy.org/en/stable/wcs/index.html
xc, yc = wcs.world_to_pixel(c1)

# plot the coutout
dx = 200
ax = plt.subplot(projection=wcs)
ax.imshow(hdu.data, vmin=0, vmax=0.005, origin='lower')
ax.grid(color='white', ls='solid')
ax.set(xlim=(xc-dx,xc+dx), ylim=(yc-dx,yc+dx))
ax.set_xlabel('Right Ascension')
ax.set_ylabel('Declination')
plt.savefig('cutout.png')
# plt.show()

iRA  = 1  # RA column
iDEC = 4  # DEC column


# search data table for entries up to 120 arcsec away from c1  (may take a while ... )
for d in data :
    # just some preselection to speed up loop
    if d[iRA] < 203.2  or d[iRA] > 207.2  :  # position of c1 205.15916667, 55.83916667
        continue
    c2 = SkyCoord(d[iRA],d[iDEC],unit=u.deg)
    sep = (c1.separation(c2)).arcsec
    if sep < 120.0 :
        print('  c1:  ', c1.ra.to_string(u.hour) + ' ' + c1.dec.to_string(u.degree) )
        print('  c2:  ', c2.ra.to_string(u.hour) + ' ' + c2.dec.to_string(u.degree) )
        print('   sep:  ', sep, '  arcsec' )



