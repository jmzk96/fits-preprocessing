import struct
from math import prod
from typing import BinaryIO, List

import numpy as np

from hda_fits.logging_config import logging
from hda_fits.types import Layout, SOMHeader

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class SOM:
    """
    A class to represent the SOM

    Attributes
    ----------
    header : SOMHeader
        header of SOM
    layout : Layout
        layout of SOM in som file
    channels : int
        depth of neuron in SOM
    nodes : List[np.ndarray]
        nodes/image data that are contained in som file after header

    Methods
    ----------
    get_node(row:int,column:int)
        gets a specific node from SOM

    """

    def __init__(self, header: SOMHeader, nodes: List[np.ndarray]):
        """
        Constructor for SOM

        Parameters
        ----------
        header :  SOMHeader
            header of SOM
        nodes : List[np.ndarray]
            nodes/image data that are contained in som file after header
        """
        self.header = header
        self.layout = header.som_layout
        self.channels = header.neuron_layout.depth
        self.nodes = nodes

    def get_node(self, row: int, column: int):
        """
        a method of SOM to get a specific node from SOM

        Parameters
        ----------
        row : int
            row of SOM which contains node. Starts with 0.
        column : int
            column of SOM which contains node. Starts with 0.

        Returns
        ----------
        numpy.ndarray
        """
        grid_offset = (row * self.layout.width) + column
        return self.nodes[grid_offset]


def read_som_file_header_from_stream(file_stream: BinaryIO) -> SOMHeader:
    """
    a function to read and obtain header information from som file. This function is used by read_file_som_header.

    Parameters
    ----------
    file_stream :  BinaryIO
        file object for som file

    Returns
    ----------
    SOMHeader
    """
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
    """
    a function that uses read_som_file_header_from_stream to read som file

    Parameters
    ----------
    filepath : str
        filepath to som file

    Returns
    ----------
    SOMHeader
    """
    with open(filepath, "rb") as file_stream:
        header = read_som_file_header_from_stream(file_stream=file_stream)

    return header


def read_som_file_node_from_stream(
    file_stream: BinaryIO,
    image_number: int,
    header_offset: int,
    layout: Layout,
) -> np.ndarray:
    """
    a function that reads a specific SOM node from som file

    Parameters
    ----------
    file_stream : BinaryIO
        file object of som file
    image_number : int
        number or index of node in som file
    header_offset :  int
        offset of header in som file
    layout : Layout
        Layout of neuron in som file

    Returns
    ----------
    numpy.ndarray
    """
    image_size = layout.width * layout.height * layout.depth
    file_stream.seek(image_size * image_number * 4 + header_offset, 0)
    data = struct.unpack("f" * image_size, file_stream.read(image_size * 4))

    if layout.depth == 1:
        image = np.array(data).reshape((layout.height, layout.width))
    else:
        image = np.array(data).reshape((layout.depth, layout.height, layout.width))

    return image


def read_som(filepath: str) -> SOM:
    """
    a function to read SOMs from som file and return SOM object

    Parameter
    ----------
    filepath : str
        filepath to som file

    Returns
    ----------
    SOM
    """
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
