from math import prod

import numpy as np

from hda_fits import map
from hda_fits.logging_config import logging
from hda_fits.types import Layout, MapHeader

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def test_read_map_file_header(test_map_file):
    header = map.read_map_file_header(test_map_file)

    assert type(header) == MapHeader
    assert header.version == 2
    assert header.file_type == 2
    assert header.data_type == 0
    assert header.number_of_images == 20
    assert header.data_layout == 0
    assert header.dimensionality == 2
    assert header.som_layout == Layout(width=2, height=2, depth=1)
    assert header.header_end_offset == 32


def test_read_map_file_mapping(test_map_file):
    image_number = 2
    mapping = map.read_map_file_mapping(test_map_file, image_number)
    header = map.read_map_file_header(test_map_file)

    assert mapping is not None
    assert mapping.shape == (header.som_layout.width, header.som_layout.height)


def test_count_images_per_class(test_map_file):
    header = map.read_map_file_header(test_map_file)
    countarray, pos_vec = map.count_images_per_class(
        header.number_of_images, header.som_layout, test_map_file
    )

    assert len(countarray) == prod(header.som_layout)
    assert len(pos_vec) == header.number_of_images
    assert type(countarray[0]) == np.int32
    assert type(pos_vec[0]) == int
    assert countarray.sum() == len(pos_vec)


def test_create_selection_vec():
    pass
