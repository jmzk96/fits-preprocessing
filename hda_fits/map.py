import struct
from math import prod
from typing import BinaryIO, List, Tuple

import numpy as np
from astropy.nddata import Cutout2D

from hda_fits.types import Layout, MapHeader


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
        data_layout,  # 0 (box) / 1 (hex)
        som_dimensionality,  # 1, 2, 3, .. -> 1D, 2D, 3D, ..
    ) = struct.unpack("i" * 6, file_stream.read(4 * 6))

    som_dimensions = struct.unpack(
        "i" * som_dimensionality, file_stream.read(4 * som_dimensionality)
    )

    number_of_images = number_of_data_entries
    som_width = som_dimensions[0]
    som_height = som_dimensions[1] if som_dimensionality > 1 else 1
    som_depth = som_dimensions[2] if som_dimensionality > 2 else 1

    som_layout = Layout(width=som_width, height=som_height, depth=som_depth)

    header_end_offset = file_stream.tell()

    return MapHeader(
        version=version,
        file_type=file_type,
        data_type=data_type,
        number_of_images=number_of_images,
        data_layout=data_layout,
        dimensionality=som_dimensionality,
        som_layout=som_layout,
        header_end_offset=header_end_offset,
    )


def read_map_file_mapping_from_stream(
    file_stream, som_size, image_number, header_offset, layout
) -> np.ndarray:
    """Read an image from an absolute position

    This function will seek to the absolute position in the open file
    and read the mapping for a specific floating point values.
    """
    file_stream.seek(som_size * image_number * 4 + header_offset, 0)
    data = struct.unpack("f" * som_size, file_stream.read(som_size * 4))

    if layout.depth == 1:
        mapping = np.array(data).reshape((layout.height, layout.width))
    else:
        mapping = np.array(data).reshape((layout.depth, layout.height, layout.width))

    return mapping


def read_map_file_mapping(filepath: str, image_number: int) -> np.ndarray:
    with open(filepath, "rb") as file_stream:
        header = read_map_file_header_from_stream(file_stream=file_stream)
        layout = header.som_layout
        som_size = layout.width * layout.height * layout.depth

        mapping = read_map_file_mapping_from_stream(
            file_stream, som_size, image_number, header.header_end_offset, layout
        )

    return mapping


def count_images_per_class(mapfile: str) -> Tuple[np.ndarray, List[int]]:

    header = read_map_file_header(mapfile)
    som_layout = header.som_layout
    number_of_images = header.number_of_images

    image_count_per_node = np.zeros(prod(som_layout), dtype=np.int32)
    node_per_image = []

    for i in range(number_of_images):
        mapping = read_map_file_mapping(mapfile, i).flatten()
        pos = int(np.argmin(mapping))
        node_per_image.append(pos)
        image_count_per_node[pos] += 1

    return image_count_per_node, node_per_image


def create_selection_index(
    node_per_image: List[int], nodes_selection_list: List[int]
) -> List[bool]:
    image_index = []

    for node in node_per_image:
        image_index.append(node not in nodes_selection_list)

    return image_index


def average_quantisation_error(filepath: str) -> float:
    header = read_map_file_header(filepath=filepath)
    list_of_mins = list()
    for i in range(header.number_of_images):
        list_of_mins.append(
            np.min(read_map_file_mapping(filepath=filepath, image_number=i))
        )
    aqe = sum(list_of_mins) / header.number_of_images
    return aqe


def topological_error(filepath: str) -> float:
    header = read_map_file_header(filepath=filepath)
    coherence_count = 0
    for i in range(header.number_of_images):
        array = read_map_file_mapping(filepath=filepath, image_number=i)
        index = np.unravel_index(np.argsort(array, axis=None), array.shape)
        first_min_index = index[0][0], index[1][0]
        second_min = array[index][1]
        cutout = Cutout2D(array, first_min_index, (3, 3)).data
        if second_min in cutout:
            coherence_count += 1
    te = coherence_count / header.number_of_images * 100
    return te
