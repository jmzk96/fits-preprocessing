import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="hda_fits",
    version="0.0.1",
    author="Projektgruppe GSI",
    author_email="dennis.imhof@stud.h-da.de",
    description="FITS-file preprocessing library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://code.fbi.h-da.de/ds-projekt-gsi-wise-2122/fits_preprocessing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
