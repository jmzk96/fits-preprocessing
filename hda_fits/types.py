from typing import Literal, NamedTuple


class WCSCoordinates(NamedTuple):
    """Right ascension (RA) and declination (DEC) coordinates"""

    RA: float
    DEC: float


class RectangleSize(NamedTuple):
    """The height and width of a rectangle"""

    image_height: int
    image_width: int


class Layout(NamedTuple):
    width: int
    height: int
    depth: int = 1


class PinkHeader(NamedTuple):
    version: Literal[1, 2]
    file_type: Literal[0]
    # Only 32 bit floating point = 0
    data_type: Literal[0]
    number_of_images: int
    # Cartesian = 0 / Hexadecimal = 1
    data_layout: Literal[0, 1]
    dimensionality: int
    layout: Layout
    header_end_offset: int


class SOMHeader(NamedTuple):
    version: Literal[1, 2]
    file_type: Literal[1]
    # Only 32 bit floating point = 0
    data_type: Literal[0]
    # Cartesian = 0 / Hexadecimal = 1
    data_layout: Literal[0, 1]
    som_dimensionality: int
    som_layout: Layout
    neuron_dimensionality: int
    # --neuron-dimension, -d <int>
    #   Dimension for quadratic SOM neurons
    #   (default = 2 * image-dimension / sqrt(2)).
    neuron_layout: Layout
    header_end_offset: int


class MapHeader(NamedTuple):
    version: Literal[1, 2]
    file_type: Literal[2]
    # Only 32 bit floating point = 0
    data_type: Literal[0]
    number_of_images: int
    # Cartesian = 0 / Hexadecimal = 1
    data_layout: Literal[0, 1]
    dimensionality: int
    som_layout: Layout
    header_end_offset: int


class SDSSFields(NamedTuple):
    run: int
    cam_col: int
    field: int

    def __str__(self):
        return f"run-{self.run}-camCol-{self.cam_col}-field-{self.field}"
