import numpy as np

from hda_fits import image_processing as himg


def test_max_weight_in_middle():
    image = np.random.rand(23, 23)
    weighted_image_radius = himg.create_weight_factors_euclidean_radius(image)
    one_over_r = weighted_image_radius / image
    assert np.unravel_index(np.argmax(one_over_r, axis=None), one_over_r.shape) == (
        11,
        11,
    )
    weighted_image_gauss = himg.create_weight_factors_gauss(image,std=5.0)
    gauss = weighted_image_gauss / image
    assert np.unravel_index(np.argmax(gauss, axis=None), gauss.shape) == (11, 11)

    image = np.random.rand(12, 12)
    weighted_image_radius = himg.create_weight_factors_euclidean_radius(image)
    one_over_r = weighted_image_radius / image
    assert np.unravel_index(np.argmax(one_over_r, axis=None), one_over_r.shape) == (
        6,
        6,
    )
    weighted_image_gauss = himg.create_weight_factors_gauss(image,std=5.0)
    gauss = weighted_image_gauss / image
    assert np.unravel_index(np.argmax(gauss, axis=None), gauss.shape) == (6, 6)
