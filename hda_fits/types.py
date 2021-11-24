from typing import Literal, NamedTuple


class WCSCoordinates(NamedTuple):
    """Right ascension (RA) and declination (DEC) coordinates"""

    RA: float
    DEC: float


class RectangleSize(NamedTuple):
    """The height and width of a rectangle"""

    image_height: int
    image_width: int


class PinkLayout(NamedTuple):
    width: int
    height: int
    depth: int


class PinkHeader(NamedTuple):
    version: Literal[1, 2]
    file_type: Literal[0]
    # Only 32 bit floating point = 0
    data_type: Literal[0]
    number_of_images: int
    # Cartesian = 0 / Hexadecimal = 1
    data_layout: Literal[0, 1]
    dimensionality: int
    layout: PinkLayout
    header_end_offset: int
