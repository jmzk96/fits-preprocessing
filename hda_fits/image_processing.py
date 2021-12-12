import numpy as np

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


def create_masked_border(image, border_proportion: float):
    mask = np.zeros(image.shape)
    top, right, bottom, left = calculate_border_coordinates(image, border_proportion)
    mask[top:bottom, left:right] = image[top:bottom, left:right]
    return mask


def calculate_bounding_box(image, factor_std, padding, border_proportion=0.05):

    image_masked = create_masked_border(image, border_proportion=border_proportion)
    xy = np.argwhere(
        image_masked > (image_masked.std() * factor_std + image_masked.mean())
    )
    x = xy[:, 0]
    y = xy[:, 1]

    left, right = np.min(y) - padding, np.max(y) + padding
    bottom, top = np.max(x) + padding, np.min(x) - padding

    return BoxCoordinates(top=top, right=right, bottom=bottom, left=left)
