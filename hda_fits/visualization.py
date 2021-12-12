from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from hda_fits import image_processing as himg
from hda_fits import map as hmap
from hda_fits.som import SOM


def show_som(
    som: SOM, channel: int = 0, figsize: Tuple[int, int] = (12, 12), cmap: str = "jet"
) -> Tuple[Figure, Axes]:
    """
    Function to visualize the SOM nodes in its
    respective grid layout.

    Will currently only support 2D-layouts
    """
    width, height, depth = som.layout
    fig, axes = plt.subplots(height, width, figsize=figsize)
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            ax.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            node = som.get_node(i, j)

            if depth > 1:
                ax.imshow(node[channel, :, :], cmap=cmap)
            else:
                ax.imshow(node, cmap=cmap)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    return fig, axes


def show_merged_som(
    som: SOM, figsize: Tuple[int, int] = (12, 12), cmap: str = "jet"
) -> Tuple[Figure, Axes]:
    """
    Function to visualize the SOM nodes in its
    respective grid layout.

    Will currently only support 2D-layouts
    """
    width, height, number_of_channels = som.layout

    fig, axes = plt.subplots(height, width, figsize=figsize)
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            ax.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            node = som.get_node(i, j)
            image_radio = node[0, :, :]
            image_optical = node[1, :, :]

            ax.imshow(image_optical, cmap="jet")
            ax.contour(image_radio, alpha=0.95)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    return fig, axes


def show_merged_som_node(
    som: SOM, row: int, col: int, figsize: Tuple[int, int] = (12, 12)
) -> Tuple[Figure, Axes]:

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    node = som.get_node(row, col)
    image_radio = node[0, :, :]
    image_optical = node[1, :, :]
    ax.imshow(image_optical, cmap="jet")
    ax.contour(image_radio, alpha=0.95)

    return fig, ax


def show_count_heatmap(
    filepath_map: str, figsize: Tuple[int, int]
) -> Tuple[Figure, Axes]:

    header = hmap.read_map_file_header(filepath_map)
    width, height, _ = header.som_layout

    image_count_per_node, node_per_image = hmap.count_images_per_class(filepath_map)
    ic = image_count_per_node.reshape((height, width))
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(ic.astype("int"), annot=True, fmt="d", cmap="jet")

    return fig, ax


def show_snr_histogram(
    filepath_pink: str, figsize: Tuple[int, int] = (12, 8), bins: int = None
) -> Tuple[Figure, Axes]:
    snr = himg.calculate_snrs_on_pink_file(filepath_pink)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(snr, bins=bins)

    return fig, ax


def show_bounding_box(
    image, factor_std=2, padding=0, border_proportion=0.05
) -> Tuple[Figure, Axes]:
    radio = image[0]
    top, right, bottom, left = himg.calculate_bounding_box(
        radio, factor_std, padding, border_proportion
    )

    fig, ax = plt.subplots(1, 1)
    ax.imshow(radio)
    ax.axhline(y=bottom, color="white")
    ax.axhline(y=top, color="white")
    ax.axvline(x=left, color="white")
    ax.axvline(x=right, color="white")

    return fig, ax
