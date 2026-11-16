"""Plaintext HOG stages and classical comparison; encrypted specification: hog_fhe.py."""

import numpy as np

IMAGE_SHAPE = (16, 16)
IMAGE_SIZE = 256
CELL_SIZE = 8
BINS = 9
FEATURES = 4 * BINS
STAGES = ("grad_x", "grad_y", "magnitude_sq", "histogram_poly", "descriptor_poly")
ANGLES = np.arange(BINS) * np.pi / BINS
VOTE_SCALE = 8 / (3 * BINS)
NORM_EPS = 0.25
# Our normalization polynomial covers sum(h*h) in [0,35/648] for any allowed image.
NORM_SUMSQ_MAX = 35 / 648
NORM_CENTER = NORM_EPS**2 + NORM_SUMSQ_MAX / 2


def make_input(n=IMAGE_SIZE):
    if n != IMAGE_SIZE:
        raise ValueError("HOG expects 256 pixels (16x16).")
    rows, cols = np.indices(IMAGE_SHAPE)
    blob = np.exp(-((rows - 5)**2 + (cols - 5)**2) / 8)
    square = (rows >= 9) & (rows < 13) & (cols >= 9) & (cols < 13)
    noise = np.random.default_rng(0).random(IMAGE_SHAPE)
    # Match SURF's synthetic image so the image-kernel experiments share an input.
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


def _pack(features):
    result = np.zeros(IMAGE_SIZE, dtype=features.dtype)
    # Four cells times nine bins fill slots 0:36; the harness expects 256 outputs.
    result[:FEATURES] = features.ravel()
    return result


def gradients(signal):
    image = as_image(signal)
    gx, gy = np.zeros_like(image), np.zeros_like(image)
    # Dalal-Triggs, Sec. 6.2: centered differences with the [-1,0,1] derivative mask.
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy[1:-1, :] = image[2:, :] - image[:-2, :]
    # Undefined normal derivatives stay zero at the outer rows and columns.
    return gx, gy


def polynomial_histograms(signal):
    gx, gy = gradients(signal)
    angles = ANGLES.astype(gx.dtype)
    # Our CKKS variant replaces Dalal's magnitude/angle votes with smooth cos^4 energy.
    projection = (gx[..., None] * np.cos(angles) + gy[..., None] * np.sin(angles)) / 2
    square = projection * projection
    votes = square * square * np.asarray(VOTE_SCALE, dtype=gx.dtype)
    # Dalal-Triggs, Sec. 6.3: collect nine unsigned directions in each 8x8 cell.
    return votes.reshape(2, CELL_SIZE, 2, CELL_SIZE, BINS).mean(axis=(1, 3))


def inverse_sqrt_polynomial(sumsq):
    s = np.asarray(sumsq)
    t = (s - np.asarray(NORM_SUMSQ_MAX / 2, dtype=s.dtype)) / np.asarray(
        NORM_CENTER, dtype=s.dtype)
    # Our quadratic approximates 1/sqrt(eps^2+s); its relative error is below 0.98%.
    return (1 - t / 2 + 3 * t * t / 8) / np.asarray(np.sqrt(NORM_CENTER), dtype=s.dtype)


def features(signal, stage="grad_x"):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}.")
    if stage in ("grad_x", "grad_y", "magnitude_sq"):
        gx, gy = gradients(signal)
        if stage == "grad_x":
            return gx.ravel()
        if stage == "grad_y":
            return gy.ravel()
        return (gx * gx + gy * gy).ravel()

    hist = polynomial_histograms(signal).ravel()
    if stage == "descriptor_poly":
        # Dalal's block-normalization stage is approximated without secret sqrt/division.
        hist = hist * inverse_sqrt_polynomial(np.dot(hist, hist))
    return _pack(hist)


def classical_histograms(signal, spatial_interpolation=True):
    gx, gy = gradients(signal)
    # Dalal-Triggs, Sec. 6.3: exact magnitude and unsigned angle for the comparison.
    magnitude = np.hypot(gx, gy) / 2
    angle = np.mod(np.arctan2(gy, gx), np.pi) * BINS / np.pi
    lower = np.floor(angle).astype(int)
    fraction = angle - lower
    hist = np.zeros((2, 2, BINS), dtype=gx.dtype)
    for row in range(16):
        for col in range(16):
            cells = [(row // CELL_SIZE, col // CELL_SIZE, 1.0)]
            if spatial_interpolation:
                cell_y = (row + 0.5) / CELL_SIZE - 0.5
                cell_x = (col + 0.5) / CELL_SIZE - 0.5
                y0, x0 = int(np.floor(cell_y)), int(np.floor(cell_x))
                cells = []
                for y in (y0, y0 + 1):
                    for x in (x0, x0 + 1):
                        if 0 <= y < 2 and 0 <= x < 2:
                            weight = (1 - abs(cell_y - y)) * (1 - abs(cell_x - x))
                            cells.append((y, x, weight))
            for y, x, weight in cells:
                # Match the variant's public gradient gain and cell-average units.
                vote = magnitude[row, col] * weight / CELL_SIZE**2
                b, f = lower[row, col], fraction[row, col]
                hist[y, x, b % BINS] += vote * (1 - f)
                hist[y, x, (b + 1) % BINS] += vote * f
    return hist


def classical_descriptor(signal):
    hist = classical_histograms(signal).ravel()
    # Dalal-Triggs, Sec. 6.4: L2-Hys normalizes, clips at 0.2, then normalizes again.
    hist = hist / np.sqrt(np.dot(hist, hist) + 1e-10)
    hist = np.minimum(hist, 0.2)
    return _pack(hist / np.sqrt(np.dot(hist, hist) + 1e-10))


def exact_surrogate_descriptor(signal):
    hist = polynomial_histograms(signal).ravel()
    # Keep the same votes and epsilon to isolate only the normalization approximation.
    return _pack(hist / np.sqrt(np.dot(hist, hist) + NORM_EPS**2))


def approximation_errors(signal):
    histogram = features(signal, "histogram_poly")
    descriptor = features(signal, "descriptor_poly")
    return {
        "histogram_vs_classical": float(np.max(np.abs(histogram - _pack(classical_histograms(signal))))),
        "descriptor_vs_classical": float(np.max(np.abs(descriptor - classical_descriptor(signal)))),
        "normalization_only": float(np.max(np.abs(descriptor - exact_surrogate_descriptor(signal)))),
    }


if __name__ == "__main__":
    for name, error in approximation_errors(make_input()).items():
        print(f"{name}: {error:.6g}")
