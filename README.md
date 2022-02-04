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


## Nutzung

Die Nutzung der Library ist  [in den Tutorialnotebooks](tutorial_notebooks/) beschrieben.
Hier sind die Use-Cases im Rahmen der Projektarbeit nach aufsteigender Komplexität dargestellt. Weiterhin ist
es empfehlenswert sich zunächst mit der theoretisch fachlichen Abhandlung des Forschungsgegenstands vertraut zu machen
und hierzu den Projektbericht, der als PDF unter [docs/Projektbericht.pdf](docs/Projektbericht.pdf) verfügbar ist, zu lesen.

Die Tutorianotebooks beschreiben hierbei folgende Sachverhalte und Use-Cases:

### 01 - Intro, Radiobilder und PINK-Dateiformate 
[](tutorial_notebooks/01_Intro_Radiobilder_und_PINK.ipynb)

### 02 - Optische Bilder und Pan-STARRs
[](tutorial_notebooks/02_Optische_Bilder_PanSTARRS.ipynb)

### 03 - Multichannel-Bilder und Bildtransformationen
[](tutorial_notebooks/03_Transformationen_und_Multichannel_PINK.ipynb)

# Entwicklung

Es werden bei der Installation von `hda_fits` via

```
pip install .
```

oder

```
python setup.py install
```

auch die in `requirements_dev.txt` spezifizierten
Dependencies installiert. Nach der Installation der Library und zugehörigen 
Dependencies sollte weiterhin folgender Befehl ausgeführt werden:

```
pre-commit install
```

Dieser Befehl installiert die in `.pre-commit-config.yaml` spezifizierten Hooks, die
beim Ausführen von `git commit` und `git push` prüfen, ob der Code grundsätzlichen
Code-Qualitätsstandards entspricht.

Falls dem nicht so ist, bekommt man die Fehlerbeschreibung und 
die Dateien und Zeilen angezeigt, die noch verbesserungswürdig sind 
und der Commit schlägt fehl. Man muss daraufhin die Zeilen verbessern und kann
dann den Commit erfolgreich durchführen.


# Weitere Informationen

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
