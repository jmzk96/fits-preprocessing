import struct
from typing import BinaryIO

import numpy as np

from hda_fits.types import MapHeader, MapLayout


def read_map_file_header(filepath: str) -> MapHeader:
    with open(filepath, "rb") as file_stream:
        header = read_map_file_header_from_stream(file_stream=file_stream)

    return header


def read_map_file_header_from_stream(file_stream: BinaryIO) -> MapHeader:
    (
        version,  # 1 / 2
        file_type,  # 2 (Map-File)
        data_type,  # 0 (32 bit floats)
        number_of_data_entries,  # == number_of_images
        som_layout,  # 0 (box) / 1 (hex)
        som_dimensionality,  # 1, 2, 3, .. -> 1D, 2D, 3D, ..
    ) = struct.unpack("i" * 6, file_stream.read(4 * 6))

    numberOfImages = number_of_data_entries
    somWidth = som_dimensionality[0]
    somHeight = som_dimensionality[1] if som_dimensionality > 1 else 1
    somDepth = som_dimensionality[2] if som_dimensionality > 2 else 1

    map_layout = MapLayout(width=somWidth, height=somHeight, depth=somDepth)

    header_end_offset = file_stream.tell()

    return MapHeader(
        version,
        file_type,
        data_type,
        numberOfImages,
        som_layout,
        som_dimensionality,
        map_layout,
        somWidth,
        somHeight,
        somDepth,
        header_end_offset,
    )


def read_map_file_mapping_from_stream(
    file_stream,
    som_size,
    image_number,
    header_offset,
    somWidth,
    somHeight,
    somDepth,
):
    """Read an image from an absolute position

    This function will seek to the absolute position in the open file
    and read the mapping for a specific floating point values.
    """
    file_stream.seek(som_size * image_number * 4 + header_offset, 0)
    data = struct.unpack("f" * som_size, file_stream.read(som_size * 4))

    if somDepth == 1:
        mapping = np.array(data).reshape((somHeight, somWidth))
    else:
        mapping = np.array(data).reshape((somDepth, somHeight, somWidth))

    return mapping


def read_map_file_mapping(filepath: str, image_number: int) -> np.ndarray:
    with open(filepath, "rb") as file_stream:
        header = read_map_file_header_from_stream(file_stream=file_stream)
        som_size = header.somWidth * header.somHeight * header.somDepth

        mapping = read_map_file_mapping_from_stream(
            file_stream,
            som_size,
            image_number,
            header.somWidth,
            header.somHeight,
            header.somDepth,
        )

    return mapping


# def count_images_per_class(imagecount, somwidth, somheight, somdepth, mapfile):

#   if somdepth == 1:
#       countarray = np.zeros((somwidth, somheight), dtype=np.int16)
#    else:
#      countarray = np.zeros((somwidth, somheight, somdepth), dtype=np.int16)

#    for i in range(imagecount - 1):
# pos = pos_class_of_image(i, mapfile)  # Funktion ist noch zu schreiben
# countarray[pos] = +1

#   return countarray


# soll die Position im Som Array wieder geben für die das betrachtete Bild den kleinsten Floatwert annimmt
