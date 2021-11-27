from math import prod

import numpy as np
import pytest

import hda_fits as hfits
from hda_fits import som
from hda_fits.logging_config import logging
from hda_fits.types import Layout, SOMHeader

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def test_read_som_file_header(test_som_file):
    header = som.read_som_file_header(test_som_file)

    assert type(header) == SOMHeader
    assert header.version == 2
    assert header.file_type == 1
    assert header.data_type == 0
    assert header.data_layout == 0
    assert header.som_dimensionality == 2
    assert header.som_layout == Layout(depth=1, height=2, width=2)
    assert header.neuron_dimensionality == 2
    assert header.neuron_layout == Layout(depth=1, height=135, width=135)


def test_read_som(test_som_file):
    som = hfits.som.read_som(test_som_file)

    number_of_nodes = prod(som.header.som_layout)

    assert som is not None
    assert len(som.nodes) == number_of_nodes


def test_get_node(test_som_file):

    som = hfits.som.read_som(test_som_file)

    first_node = som.get_node(0, 0)
    second_node = som.get_node(0, 1)
    last_node = som.get_node(1, 1)
    assert first_node.shape == (135, 135)
    assert last_node.shape == (135, 135)
    assert np.array_equal(som.nodes[1], second_node)

    with pytest.raises(IndexError):
        som.get_node(1, 2)

    with pytest.raises(IndexError):
        som.get_node(2, 1)
