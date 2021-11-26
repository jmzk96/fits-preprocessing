import struct
from math import prod
from typing import BinaryIO, List

import numpy as np

from hda_fits.logging_config import logging
from hda_fits.types import Layout, SOMHeader

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class SOM:
    def __init__(self, header: SOMHeader, nodes: List[np.ndarray]):
        self.header = header
        self.layout = header.som_layout
        self.nodes = nodes

    def get_node(self, row: int, column: int, channel: int = 0):
        grid_offset = (row * self.layout.width) + column
        if channel == 0:
            return self.nodes[grid_offset]

        channel_offset = channel * (self.layout.width * self.layout.height)
        return self.nodes[channel_offset + grid_offset]


def read_som_file_header_from_stream(file_stream: BinaryIO) -> SOMHeader:
    (
        version,
        file_type,
        data_type,
        data_layout,
        som_dimensionality,
    ) = struct.unpack("i" * 5, file_stream.read(4 * 5))

    som_depth = (
        struct.unpack("i", file_stream.read(4))[0] if som_dimensionality > 2 else 1
    )
    som_height = (
        struct.unpack("i", file_stream.read(4))[0] if som_dimensionality > 1 else 1
    )
    som_width = struct.unpack("i", file_stream.read(4))[0]

    som_layout = Layout(width=som_width, height=som_height, depth=som_depth)

    neuron_layout, neuron_dimensionality = struct.unpack(
        "i" * 2, file_stream.read(4 * 2)
    )

    neuron_depth = (
        struct.unpack("i", file_stream.read(4))[0] if neuron_dimensionality > 2 else 1
    )
    neuron_height = (
        struct.unpack("i", file_stream.read(4))[0] if neuron_dimensionality > 1 else 1
    )
    neuron_width = struct.unpack("i", file_stream.read(4))[0]

    neuron_layout = Layout(width=neuron_width, height=neuron_height, depth=neuron_depth)

    header_end_offset = file_stream.tell()

    return SOMHeader(
        version=version,
        file_type=file_type,
        data_type=data_type,
        data_layout=data_layout,
        som_dimensionality=som_dimensionality,
        som_layout=som_layout,
        neuron_dimensionality=neuron_dimensionality,
        neuron_layout=neuron_layout,
        header_end_offset=header_end_offset,
    )


def read_som_file_header(filepath: str) -> SOMHeader:
    with open(filepath, "rb") as file_stream:
        header = read_som_file_header_from_stream(file_stream=file_stream)

    return header


def read_som_file_node_from_stream(
    file_stream: BinaryIO,
    image_number: int,
    header_offset: int,
    layout: Layout,
) -> np.ndarray:
    image_size = layout.width * layout.height * layout.depth
    file_stream.seek(image_size * image_number * 4 + header_offset, 0)
    data = struct.unpack("f" * image_size, file_stream.read(image_size * 4))

    if layout.depth == 1:
        image = np.array(data).reshape((layout.height, layout.width))
    else:
        image = np.array(data).reshape((layout.depth, layout.height, layout.width))

    return image


def read_som(filepath: str) -> SOM:
    """Reads and reshapes the data of a .pink image"""
    with open(filepath, "rb") as file_stream:
        header = read_som_file_header_from_stream(file_stream=file_stream)
        som_layout = header.som_layout
        neuron_layout = header.neuron_layout
        number_of_nodes = prod(som_layout)
        nodes = []
        for i in range(number_of_nodes):
            node = read_som_file_node_from_stream(
                file_stream, i, header.header_end_offset, neuron_layout
            )
            nodes.append(node)
    return SOM(header, nodes)
