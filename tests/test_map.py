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
