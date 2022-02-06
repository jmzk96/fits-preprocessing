from typing import Literal, NamedTuple


class WCSCoordinates(NamedTuple):
    """
    A class to represent WSCoordinates as used in astropy. Inherits from NamedTuple.

    Attributes
    ----------
    RA  : float
        Right ascension
    DEC : float
        declination
    """

    RA: float
    DEC: float


class RectangleSize(NamedTuple):
    """
    A class to represent height and width of rectangular cutout. Inherits from NamedTuple.

    Attributes
    ----------
    image_height  : int
        height of rectangular image cutout
    DEC : int
        width of rectangular image cutout
    """

    image_height: int
    image_width: int


class Layout(NamedTuple):
    """
    A class to represent layout of image or neuron or som in pink, som or map file. Inherits from NamedTuple.

    Attributes
    ----------
    width  : int
        width of image or neuron or som in pink, som or map file.
    height : int
        height of image or neuron or som in pink, som or map file.
    depth : int
        depth of image or neuron or som in pink, som or map file.
    """

    width: int
    height: int
    depth: int = 1


class BoxCoordinates(NamedTuple):
    """
    A class to represent coordinates for bounding box. Inherits from NamedTuple.

    Attributes
    ----------
    top  : int
        top coordinate of 2D bounding box.
    right : int
        right coordinate of 2D bounding box.
    bottom : int
        bottom coordinate of 2D bounding box.
    left : int
        left coordinate of 2D bounding box.
    """

    top: int
    right: int
    bottom: int
    left: int


class PinkHeader(NamedTuple):
    """
    A class to represent header information in pink file. Inherits from NamedTuple.

    Attributes
    ----------
    version  : Literal[1,2]
        file format version of pink file. Can only take in 1 or 2.
    flle_type : Literal[0]
        file type. pink:0, som:1, map:2
    data_type : Literal[0]
        0 as an indicator of 32 bit floating point. https://github.com/HITS-AIN/PINK/blob/master/FILE_FORMATS.md
    number_of_Images : int
        number of images in file.
    data_layout : Literal[0,1]
        0 for Cartesian , 1 for Hexagonal
    dimensionality : int
        number of dimensions
    layout: Layout
        layout of image
    header_end_offset : int
        end offset position of header. After the offset comes the images
    """

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
    """
    A class to represent header information in som file. Inherits from NamedTuple.

    Attributes
    ----------
    version  : Literal[1,2]
        file format version of som file. Can only take in 1 or 2.
    flle_type : Literal[1]
        file type. pink:0, som:1, map:2
    data_type : Literal[0]
        0 as an indicator of 32 bit floating point. https://github.com/HITS-AIN/PINK/blob/master/FILE_FORMATS.md
    data_layout : Literal[0,1]
        0 for Cartesian , 1 for Hexagonal
    som_dimensionality : int
        number of dimensions for som
    som_layout :  int
        layout of som
    neuron_dimensionality : int
        number of dimensions of neuron
    neuron_layout: Layout
        layout of neuron
    header_end_offset : int
        end offset position of header. After the offset comes the images
    """

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
    """
    A class to represent header information in pink file. Inherits from NamedTuple.

    Attributes
    ----------
    version  : Literal[1,2]
        file format version of map file. Can only take in 1 or 2.
    flle_type : Literal[2]
        file type. pink:0, som:1, map:2
    data_type : Literal[0]
        0 as an indicator of 32 bit floating point. https://github.com/HITS-AIN/PINK/blob/master/FILE_FORMATS.md
    number_of_Images : int
        number of images in file.
    data_layout : Literal[0,1]
        0 for Cartesian , 1 for Hexagonal
    dimensionality : int
        number of dimensions
    som_layout: Layout
        layout of som
    header_end_offset : int
        end offset position of header. After the offset comes the images
    """

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
    """
    A class to represent sdss fields when querying sdss images. Inherits from NamedTuple.

    Attributes
    ----------
    run : int
        run number of image, which specifies the specific scan
    cam_col : int
        camera column identifying scanline within the run
    field : int
        field number of image
    """

    run: int
    cam_col: int
    field: int

    def __str__(self):
        return f"run-{self.run}-camCol-{self.cam_col}-field-{self.field}"
