from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from hda_fits import image_processing as himg
from hda_fits import map as hmap
from hda_fits import pink as hpink
from hda_fits.som import SOM

from .types import BoxCoordinates


def show_som(
    som: SOM, channel: int = 0, figsize: Tuple[int, int] = (12, 12), cmap: str = "jet"
) -> Tuple[Figure, Axes]:
    """
    Function to visualize the SOM nodes in its
    respective grid layout.

    Will currently only support 2D-layouts
    """
    width, height, _ = som.layout
    depth = som.channels
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


def show_multichannel_image(
    filepath: str, image_number: int, figsize=(24, 8), vmax=1.0
) -> Tuple[Figure, Axes]:
    # header = hpink.read_pink_file_header(filepath)
    image = hpink.read_pink_file_image(filepath, image_number=image_number)

    # number_of_channels = header.layout.depth
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    axes[0].imshow(image[0, :, :], vmin=0.0, vmax=1.0, cmap="jet")
    axes[1].imshow(image[1, :, :], vmin=0.0, vmax=vmax, cmap="jet")
    axes[2].imshow(image[1, :, :], vmin=0.0, vmax=vmax, cmap="jet")
    axes[2].contour(image[0, :, :], cmap="jet")

    return fig, axes


def show_image_with_distribution_and_snr(
    image: np.ndarray, figsize: Tuple[int, int] = (14, 6)
) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(ncols=2, figsize=figsize)

    snr = himg.calculate_signal_to_noise_ratio(image)
    fig.suptitle(f"SNR: {snr}")
    ax[0].imshow(image, cmap="jet")
    ax[1].hist(image[image != 0])
    fig.tight_layout()
    return fig, ax


def show_snr_histogram(
    filepath_pink: str,
    channel: int = 0,
    figsize: Tuple[int, int] = (12, 8),
    bins: int = None,
) -> Tuple[Figure, Axes]:
    snr = himg.calculate_snrs_on_pink_file(filepath_pink, channel)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(snr, bins=bins)

    return fig, ax


def create_axes_with_border_lines(
    image: np.ndarray, border_coordinates: BoxCoordinates, ax: Axes
) -> Axes:
    top, right, bottom, left = border_coordinates

    ax.imshow(image, cmap="jet")
    ax.axhline(y=bottom, color="white")
    ax.axhline(y=top, color="white")
    ax.axvline(x=left, color="white")
    ax.axvline(x=right, color="white")

    return ax


def show_bounding_box(
    image, factor=2, padding=0, border_proportion=0.05, figsize=(12, 12)
) -> Tuple[Figure, Axes]:
    border_coordinates = himg.calculate_bounding_box(
        image, factor, padding, border_proportion
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = create_axes_with_border_lines(image, border_coordinates, ax)
    return fig, ax


def show_bounding_box_2x2(
    image_optical,
    image_radio,
    factor_std=2,
    padding=0,
    border_proportion=0.05,
    fill_with: Union[float, Callable] = np.mean,
    figsize: Tuple[int, int] = (16, 16),
    vmax: float = 0.05,
) -> Tuple[Figure, Axes]:

    border_coordinates_proportion = himg.calculate_border_coordinates(
        image_radio, border_proportion
    )

    border_coordinates = himg.calculate_bounding_box(
        image_radio,
        factor_std=factor_std,
        padding=padding,
        border_proportion=border_proportion,
    )

    image_optical_masked = himg.create_masked_image(
        image_optical, border_coordinates, fill_with=fill_with
    )

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    for ax_big in axes:
        for ax in ax_big:
            ax.axis("off")

    create_axes_with_border_lines(
        image_radio, border_coordinates_proportion, axes[0][0]
    )
    create_axes_with_border_lines(image_radio, border_coordinates, axes[0][1])

    axes[1][0].imshow(image_optical, vmin=0.0, vmax=vmax, cmap="jet")
    axes[1][1].imshow(image_optical_masked, vmin=0.0, vmax=vmax, cmap="jet")

    return fig, axes
