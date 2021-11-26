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
    som_width = som_dimensionality[0]
    som_height = som_dimensionality[1] if som_dimensionality > 1 else 1
    som_depth = som_dimensionality[2] if som_dimensionality > 2 else 1

    map_layout = MapLayout(width=som_width, height=som_height, depth=som_depth)

    header_end_offset = file_stream.tell()

    return MapHeader(
        version,
        file_type,
        data_type,
        numberOfImages,
        som_layout,
        som_dimensionality,
        map_layout,
        som_width,
        som_height,
        som_depth,
        header_end_offset,
    )


def read_map_file_mapping_from_stream(
    file_stream,
    som_size,
    image_number,
    header_offset,
    som_width,
    som_height,
    som_depth,
):
    """Read an image from an absolute position

    This function will seek to the absolute position in the open file
    and read the mapping for a specific floating point values.
    """
    file_stream.seek(som_size * image_number * 4 + header_offset, 0)
    data = struct.unpack("f" * som_size, file_stream.read(som_size * 4))

    if som_depth == 1:
        mapping = np.array(data).reshape((som_height, som_width))
    else:
        mapping = np.array(data).reshape((som_depth, som_height, som_width))

    return mapping


def read_map_file_mapping(filepath: str, image_number: int) -> np.ndarray:
    with open(filepath, "rb") as file_stream:
        header = read_map_file_header_from_stream(file_stream=file_stream)
        som_size = header.som_width * header.som_height * header.som_depth

        mapping = read_map_file_mapping_from_stream(
            file_stream,
            som_size,
            image_number,
            header.som_width,
            header.som_height,
            header.som_depth,
        )

    return mapping


def count_images_per_class(imagecount, som_width, som_height, som_depth, mapfile):

    countarray = np.zeros(som_width * som_height * som_depth, dtype=np.int32)
    pos_vec = []

    for i in range(imagecount):
        mapping = read_map_file_mapping(mapfile, i).flatten()
        pos = np.argmin(mapping)
        pos_vec.append(pos)
        countarray[pos] += 1

    return countarray, pos_vec
