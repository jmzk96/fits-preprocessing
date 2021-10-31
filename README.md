# `hda_fits`-Library - Preprocessing von FITS und PINK-Daten

Diese Library dient zum Einlesen, Downloaden und Vorverarbeiten von
`FITS`-Dateien sowie zum Transformieren dieser in das von `PINK`
erwartete, binäre Dateiformat.

Die Library kann mit 
```sh
pip install .
``` 

installiert werden und lässt sich nun durch 

```
import hda_fits
```

importieren.

## Beispiel Use-Case

Ein vollständiger Use-Case aus

* Einlesen des Shimwell-Datenkatalogs
* Filtern des Datenkatalogs nach spezifischem Mosaic
* Einlesen eines Mosaic-Files
* Speichern von mehreren Cutouts als `PINK`-Binärdatei

kann wie folgt durchgeführt werden.

```python
import hda_fits as hf


# Laden des Shimwell-Katalogs mit reduzierter Spaltenanzahl
#   Spalten:  ["Source_Name", "RA", "DEC", "Mosaic_ID"]
table = hf.read_shimwell_catalog("data/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.fits", reduced=True)

# Filtern des Pandas DF nach Rows, die zu Mosaic P205+55 gehören
table = (
  table[table.Mosaic_ID.str.contains("P205")]
  .loc[:, ["Source_Name", "RA", "DEC", "Mosaic_ID"]]
)

# Mosaic Einlesen
hdu = hf.load_mosaic("P205+55", "../data/")[0]

# Koordinaten der Himmelsobjekte in Liste speichern
coordinates = table.loc[:, ["RA", "DEC"]].values.tolist()

# Bildausschnitte mit Dimension 200x200px der Koordinaten aus Mosaic
# ausschneiden und in kartesischem PINK-Format v2 speichern
hf.write_mosaic_objects_to_pink_file_v2(
    filepath="data/P205+55_pink_v2.bin",
    hdu=hdu,
    coordinates=coordinates,
    image_size=200  # RectangleSize(200, 200)
)
```



## Mosaic-Daten `P200+55` bezogen von

https://lofar-surveys.org/dr1_mosaics.html

## Kompletter Source Catalog (Shimwell) des Data release 1

https://lofar-surveys.org/releases.html


# Astropy und FITS-Files

Docs:
https://docs.astropy.org/en/stable/index.html
https://docs.astropy.org/en/stable/io/fits/index.html
https://learn.astropy.org/FITS-images.html
https://learn.astropy.org/rst-tutorials/FITS-cubes.html


Analyzing FITS data with Jupyter Notebooks, Python, and Astropy:
https://www.youtube.com/watch?v=n5clfJArYrU


Priya Hasan - AstroPy:
https://www.youtube.com/watch?v=29KFI0_PgoE
https://github.com/vkaustubh/astropy-tutorial


Python for Astronomy 3: Handling FITS files using Python
https://www.youtube.com/watch?v=goH9yXu4jWw
https://github.com/HorizonIITM/PythonForAstronomy
Playlist: https://www.youtube.com/watch?v=HfYR0uwYAyM&list=PL2rHXmvrOZXbQviht65mZSOW0_s1ZPpLA