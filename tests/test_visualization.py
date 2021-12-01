import hda_fits.visualization as hviz
from hda_fits.logging_config import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def test_show_som(example_som):
    fig, axes = hviz.show_som(example_som, figsize=(8, 4))

    assert axes.shape == (2, 2)
    assert fig.get_figwidth() == 8.0
    assert fig.get_figheight() == 4.0
