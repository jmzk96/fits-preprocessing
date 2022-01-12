from typing import Callable, List, Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import disk, polygon

from hda_fits import pink as hpink

from .logging_config import logging
from .types import BoxCoordinates

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def denoise_cutouts_from_above(cutout_flatarray, sigma=1.5):
    std = np.std(cutout_flatarray)
    max = np.amax(cutout_flatarray)
    threshold = max - sigma * std
    cutout_flatarray[cutout_flatarray < threshold] = 0.0
    return cutout_flatarray


def denoise_cutouts_from_mean(cutout_flatarray, sigma=1.5):
    std = np.std(cutout_flatarray)
    mean = np.mean(cutout_flatarray)
    othreshold = mean + sigma * std
    output_array = cutout_flatarray.copy()
    output_array[output_array < othreshold] = 0.0
    return output_array


def min_max(data: np.ndarray):
    dmax, dmin = data.max(), data.min()
    return (data - dmin) / (dmax - dmin)


def log_scale(data: np.ndarray, eps: float = 0.001):
    min_scaled_array = data - data.min() + eps
    return np.log(min_scaled_array)


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


def create_and_apply_image_mask(
    image_for_mask_creation: np.ndarray,
    image_to_be_masked: np.ndarray,
    factor_std: float,
    padding: int,
    border_proportion: float = 0.05,
    fill_with: Union[float, Callable] = np.mean,
) -> np.ndarray:
    border_coordinates = calculate_bounding_box(
        image_to_be_masked,
        factor_std=factor_std,
        padding=padding,
        border_proportion=border_proportion,
    )
    image_optical_masked = create_masked_image(
        image_for_mask_creation, border_coordinates, fill_with=fill_with
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


def calculate_convex_hull_coordinates(
    image: np.ndarray,
    border_proportion: float = 0.1,
    factor_std_convex_hull: float = 5,
    factor_std_border: float = 5,
    padding: int = 5,
    fill_with: Union[float, Callable] = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    image_radio_masked = create_and_apply_image_mask(
        image,
        image,
        factor_std=factor_std_border,
        padding=padding,
        border_proportion=border_proportion,
        fill_with=fill_with,
    )

    points_xy = np.argwhere(
        image_radio_masked
        > (
            image_radio_masked.mean()
            + factor_std_convex_hull * image_radio_masked.std()
        )
    )
    hull = ConvexHull(points_xy)

    hull_vertices = points_xy[hull.vertices]
    rr, cc = polygon(hull_vertices[:, 0], hull_vertices[:, 1], image.shape)

    return rr, cc


def create_masked_image_convex_hull(
    image_for_mask_creation: np.ndarray,
    image_to_be_masked: np.ndarray,
    border_proportion: float = 0.1,
    factor_std_convex_hull: float = 5,
    factor_std_border: float = 5,
    padding: int = 5,
    fill_with: Union[float, Callable] = 0.0,
) -> np.ndarray:
    rr, cc = calculate_convex_hull_coordinates(
        image_for_mask_creation,
        border_proportion=border_proportion,
        factor_std_convex_hull=factor_std_convex_hull,
        factor_std_border=factor_std_border,
        padding=padding,
        fill_with=fill_with,
    )

    if isinstance(fill_with, float):
        image_masked = np.ones(image_for_mask_creation.shape) * fill_with
    else:
        image_masked = np.ones(image_for_mask_creation.shape) * fill_with(
            image_to_be_masked
        )

    image_masked[rr, cc] = image_to_be_masked[rr, cc]

    return image_masked


def calculate_maximum_distance(
    convex_hull_coordinates: Tuple[np.ndarray, np.ndarray],
    point: Union[Tuple[float, float], List[float]],
) -> float:
    x, y = convex_hull_coordinates
    dist = np.square(x - point[0]) + np.square(y - point[1])
    return np.sqrt(np.max(dist))


def create_circular_masked_image_from_convex_hull(
    image_for_mask_creation: np.ndarray,
    image_to_be_masked: np.ndarray,
    border_proportion: float = 0.1,
    factor_std_convex_hull: float = 5,
    factor_std_border: float = 5,
    padding: int = 5,
    fill_with: Union[float, Callable] = 0.0,
) -> np.ndarray:
    convex_hull_coordinates = calculate_convex_hull_coordinates(
        image_for_mask_creation,
        border_proportion=border_proportion,
        factor_std_convex_hull=factor_std_convex_hull,
        factor_std_border=factor_std_border,
        padding=padding,
        fill_with=fill_with,
    )

    assert image_for_mask_creation.shape == image_to_be_masked.shape

    center = [dim / 2 for dim in image_to_be_masked.shape]
    radius = calculate_maximum_distance(
        convex_hull_coordinates=convex_hull_coordinates, point=center
    )
    disk_coordinates = disk(center=center, radius=radius)

    if isinstance(fill_with, float):
        image_masked = np.ones(image_to_be_masked.shape) * fill_with
    else:
        image_masked = np.ones(image_to_be_masked.shape) * fill_with(image_to_be_masked)

    image_masked[disk_coordinates] = image_to_be_masked[disk_coordinates]

    return image_masked
