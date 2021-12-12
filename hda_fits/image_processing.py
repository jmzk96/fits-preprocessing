from typing import Callable, Tuple, Union

import numpy as np

from hda_fits import pink as hpink

from .logging_config import logging
from .types import BoxCoordinates

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def calculate_signal_to_noise_ratio(image: np.ndarray) -> float:
    signal = image.mean()
    noise = image.std()
    return signal / noise


def calculate_border_coordinates(
    image: np.ndarray, border_proportion: float
) -> BoxCoordinates:
    h, w = image.shape
    left, right = int(w * border_proportion), int(w - w * border_proportion)
    top, bottom = int(h * border_proportion), int(h - h * border_proportion)
    return BoxCoordinates(top=top, right=right, bottom=bottom, left=left)


def create_masked_image(
    image: np.ndarray,
    mask_coordinates: BoxCoordinates,
    fill_with: Union[float, Callable] = 0.0,
) -> np.ndarray:

    if not isinstance(fill_with, float):
        fill_with = fill_with(image)

    image_masked = np.ones(image.shape) * fill_with
    top, right, bottom, left = mask_coordinates
    image_masked[top:bottom, left:right] = image[top:bottom, left:right]
    return image_masked


def create_masked_border(
    image, border_proportion: float
) -> Tuple[np.ndarray, BoxCoordinates]:
    coordinates = calculate_border_coordinates(image, border_proportion)
    image_masked = create_masked_image(image, coordinates)
    return image_masked, coordinates


def calculate_bounding_box(
    image, factor_std, padding, border_proportion=0.05
) -> BoxCoordinates:

    image_masked, coordinates_proportion = create_masked_border(
        image, border_proportion=border_proportion
    )

    xy = np.argwhere(
        image_masked > (image_masked.std() * factor_std + image_masked.mean())
    )

    if xy.size == 0:
        return coordinates_proportion

    x = xy[:, 0]
    y = xy[:, 1]

    left, right = np.min(y) - padding, np.max(y) + padding
    bottom, top = np.max(x) + padding, np.min(x) - padding

    h, w = image.shape

    left = left if left > 0 else 0
    right = right if right < (w - 1) else (w - 1)
    top = top if top > 0 else 0
    bottom = bottom if bottom < (h - 1) else (h - 1)

    return BoxCoordinates(top=top, right=right, bottom=bottom, left=left)


def create_masked_optical_image(
    image_optical,
    image_radio,
    factor_std,
    padding,
    border_proportion=0.05,
    fill_with: Union[float, Callable] = np.mean,
) -> np.ndarray:
    border_coordinates = calculate_bounding_box(
        image_radio,
        factor_std=factor_std,
        padding=padding,
        border_proportion=border_proportion,
    )
    image_optical_masked = create_masked_image(
        image_optical, border_coordinates, fill_with=fill_with
    )
    return image_optical_masked


def calculate_snrs_on_pink_file(filepath_pink: str, channel: int = 0) -> np.ndarray:
    header = hpink.read_pink_file_header(filepath_pink)
    channels = header.layout.depth

    number_of_images = header.number_of_images
    snrs = np.empty(number_of_images)

    for i in range(number_of_images):
        image = hpink.read_pink_file_image(filepath_pink, i)
        image = image if channels == 1 else image[channel, :, :]
        snrs[i] = calculate_signal_to_noise_ratio(image)

    return snrs
