"""Plaintext SURF response maps; encrypted specification: surf_fhe.py."""

import numpy as np
from scipy.signal import convolve2d

IMAGE_SHAPE = (16, 16)
IMAGE_SIZE = 256
FILTER_SIZES = (9, 15, 21)
DXY_WEIGHT = 0.9


def make_input(n=IMAGE_SIZE):
    if n != IMAGE_SIZE:
        raise ValueError("SURF expects 256 pixels (16x16).")
    rows, cols = np.indices(IMAGE_SHAPE)
    blob = np.exp(-((rows - 5)**2 + (cols - 5)**2) / 8)
    square = (rows >= 9) & (rows < 13) & (cols >= 9) & (cols < 13)
    noise = np.random.default_rng(0).random(IMAGE_SHAPE)
    # The same blob, square and seeded noise are used for every benchmark backend.
    return (0.1 + 0.6 * blob + 0.2 * square + 0.05 * noise).ravel()


def as_image(signal):
    x = np.asarray(signal)
    if x.shape != (IMAGE_SIZE,) or np.iscomplexobj(x):
        raise ValueError("Expected a real image flattened to shape (256,).")
    if x.dtype not in (np.dtype("float32"), np.dtype("float64")):
        x = x.astype(np.float64)
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError("Pixel intensities must be finite and in [0, 1].")
    return x.reshape(IMAGE_SHAPE)


def filter_vectors(filter_size=9):
    # Bay et al., Sec. 3: grow the box lobes to approximate Gaussian second derivatives.
    if filter_size not in FILTER_SIZES:
        raise ValueError(f"filter_size must be one of {FILTER_SIZES}.")
    lobe, radius = filter_size // 3, filter_size // 2
    band = np.zeros(filter_size)
    band[radius - lobe + 1:radius + lobe] = 1
    second = np.ones(filter_size)
    second[lobe:2 * lobe] = -2
    mixed = np.zeros(filter_size)
    mixed[radius - lobe:radius] = 1
    mixed[radius + 1:radius + lobe + 1] = -1
    return band, second, mixed


def box_filters(filter_size=9):
    band, second, mixed = filter_vectors(filter_size)
    # Normalize each Hessian filter by its full support area, including zero entries.
    area = filter_size**2
    return (np.outer(band, second) / area, np.outer(second, band) / area,
            np.outer(mixed, mixed) / area)


def hessian_components(signal, filter_size=9):
    image = as_image(signal)
    # Zero extension matches the encrypted masks; no pixel wraps across an image edge.
    return tuple(convolve2d(image, mask.astype(image.dtype), mode="same",
                            boundary="fill", fillvalue=0)
                 for mask in box_filters(filter_size))


def hessian_response(signal, filter_size=9, response="det"):
    dxx, dyy, dxy = hessian_components(signal, filter_size)
    if response == "trace":
        return (dxx + dyy).ravel()
    if response != "det":
        raise ValueError("response must be 'det' or 'trace'.")
    # Bay et al., Eq. 2: det(H) = Dxx*Dyy - (0.9*Dxy)^2.
    weighted_dxy = dxy * np.asarray(DXY_WEIGHT, dtype=dxy.dtype)
    return (dxx * dyy - weighted_dxy * weighted_dxy).ravel()


def scale_space(signal):
    # Shortell, Sec. 5: compute response maps before comparing candidate keypoints.
    return np.stack([hessian_response(signal, size).reshape(IMAGE_SHAPE)
                     for size in FILTER_SIZES])


def keypoint_candidates(responses, threshold=1e-4, border=1):
    """Return (filter_size, row, column, score) candidates after decryption."""
    maps = np.asarray(responses, dtype=float)
    if maps.shape != (len(FILTER_SIZES), *IMAGE_SHAPE) or not np.isfinite(maps).all():
        raise ValueError("Expected three finite 16x16 determinant maps, ordered 9,15,21.")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative.")
    if not isinstance(border, int) or border < 1:
        raise ValueError("border must be a positive integer.")
    candidates = []
    # Bay et al., Sec. 3: retain strict 3x3x3 maxima; subpixel interpolation is omitted.
    for row in range(border, 16 - border):
        for col in range(border, 16 - border):
            neighbors = maps[:, row - 1:row + 2, col - 1:col + 2].copy().ravel()
            score = neighbors[13]
            neighbors[13] = -np.inf
            if score > threshold and score > neighbors.max():
                candidates.append((FILTER_SIZES[1], row, col, float(score)))
    return candidates


if __name__ == "__main__":
    image = make_input()
    response = hessian_response(image).reshape(IMAGE_SHAPE)
    print("SURF determinant range:", response.min(), response.max())
    print("Candidate keypoints:", keypoint_candidates(scale_space(image)))
