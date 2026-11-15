import numpy as np


def real_fft(signal):
    """
    Orthonormal FFT of a real signal, packed into n real numbers ("halfcomplex" layout,
    as in FFTW):

        out[k]     = Re X[k],  k = 0 .. n/2
        out[n - k] = Im X[k],  k = 1 .. n/2 - 1

    The other half of the spectrum is the complex conjugate (X[n-k] = conj X[k]), so these
    n numbers hold all of it. Orthonormal scaling (divide by sqrt(n)) keeps |out| the same
    size as |signal|.

    Parameters
    ----------
    signal : array_like
        Real input signal of even length n.

    Returns
    -------
    out : ndarray
        Length-n real array in halfcomplex layout.
    """
    x = np.asarray(signal, dtype=float)
    n = len(x)
    X = np.fft.rfft(x, norm="ortho")
    out = np.zeros(n)
    out[:n // 2 + 1] = X.real
    out[n // 2 + 1:] = X.imag[1:n // 2][::-1]
    return out


def magnitude_spectrum(halfcomplex):
    """|X[k]| for k = 0 .. n/2 from a real_fft() result."""
    y = np.asarray(halfcomplex, dtype=float)
    n = len(y)
    re = y[:n // 2 + 1]
    im = np.zeros(n // 2 + 1)
    im[1:n // 2] = y[n - 1:n // 2:-1]
    return np.hypot(re, im)


if __name__ == "__main__":
    fs = 1000.0
    t = np.arange(256) / fs
    sig = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    mag = magnitude_spectrum(real_fft(sig))
    freqs = np.arange(len(mag)) * fs / len(sig)
    top = np.argsort(mag)[-2:][::-1]
    print("strongest bins (Hz):", freqs[top])
