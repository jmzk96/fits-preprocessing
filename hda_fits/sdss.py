import urllib

import requests


def getSDSSfields(ra, dec, size):  # all in degree

    fmt = "csv"
    default_url = "http://skyserver.sdss3.org/public/en/tools/search/x_sql.aspx"
    delta = 0.5 * size + 0.13
    ra_max = ra + 1.5 * delta
    ra_min = ra - 1.5 * delta
    dec_max = dec + delta
    dec_min = dec - delta

    querry = "SELECT fieldID, run, camCol, field, ra, dec, run, rerun FROM Field "
    querry += (
        "WHERE ra BETWEEN "
        + str(ra_min)
        + " and "
        + str(ra_max)
        + " and dec BETWEEN "
        + str(dec_min)
        + " and "
        + str(dec_max)
    )
    params = urllib.parse.urlencode({"cmd": querry, "format": fmt})
    url_opened = urllib.request.urlopen(default_url + "?%s" % params)
    lines = url_opened.readlines()
    return lines


def getSDSSfiles(fieldInfo, band, filepath):

    run = str(fieldInfo[0])
    camcol = str(fieldInfo[1])
    field = str(fieldInfo[2])

    fileName = (
        "frame-" + band + "-"
        "{0:06d}".format(int(run))
        + "-"
        + camcol
        + "-"
        + "{0:04d}".format(int(field))
        + ".fits.bz2"
    )
    # filename = 'frame-'+band + '-''{0:06d}'.format(int(run))+'-'+camcol+'-'+'{0:04d}'.format(int(field))+'.fits'
    http = "https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/"

    http += run + "/"
    http += camcol + "/"
    http += fileName
    with open(filepath, "wb") as f:
        content = requests.get(http).content
        f.write(content)
