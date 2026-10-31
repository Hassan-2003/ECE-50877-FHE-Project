
import numpy as np
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt


def fir_lowpass(signal, cutoff, sample_rate, num_taps=101):
    """
    Apply a low-pass FIR filter.

    Parameters
    ----------
    signal : array_like
        Input signal.
    cutoff : float
        Cutoff frequency in Hz.
    sample_rate : float
        Sampling frequency in Hz.
    num_taps : int
        Number of FIR filter coefficients.

    Returns
    -------
    filtered : ndarray
        Filtered signal.
    """
    coefficients = firwin(
        num_taps,
        cutoff,
        fs=sample_rate,
        pass_zero="lowpass"
    )

    return lfilter(coefficients, 1.0, signal)


def fir_highpass(signal, cutoff, sample_rate, num_taps=101):
    """
    Apply a high-pass FIR filter.
    """
    coefficients = firwin(
        num_taps,
        cutoff,
        fs=sample_rate,
        pass_zero="highpass"
    )

    return lfilter(coefficients, 1.0, signal)


def fir_bandpass(signal, low_cutoff, high_cutoff,
                 sample_rate, num_taps=101):
    """
    Apply a band-pass FIR filter.
    """
    coefficients = firwin(
        num_taps,
        [low_cutoff, high_cutoff],
        fs=sample_rate,
        pass_zero="bandpass"
    )

    return lfilter(coefficients, 1.0, signal)



def main():
    fs = 1000
    duration = 1.0
    t = np.arange(0, duration, 1/fs)

    # Generate input signal with three frequency components
    signal = (
        np.sin(2 * np.pi * 50 * t)
        + np.sin(2 * np.pi * 150 * t)
        + np.sin(2 * np.pi * 300 * t)
    )

    # Apply FIR filters
    lowpass = fir_lowpass(signal, 100, fs)
    highpass = fir_highpass(signal, 200, fs)
    bandpass = fir_bandpass(signal, 100, 200, fs)

    # Calculate frequency spectra
    def magnitude_spectrum(x):
        spectrum = np.fft.rfft(x)
        frequencies = np.fft.rfftfreq(len(x), 1/fs)

        magnitude = 20 * np.log10(
            np.abs(spectrum) / len(x) + 1e-12
        )

        return frequencies, magnitude

    signals = [
        ("Original Signal", signal),
        ("Low-Pass (100 Hz)", lowpass),
        ("High-Pass (200 Hz)", highpass),
        ("Band-Pass (100-200 Hz)", bandpass)
    ]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    for ax, (name, x) in zip(axes, signals):
        frequencies, magnitude = magnitude_spectrum(x)

        ax.plot(frequencies, magnitude)
        ax.set_title(name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_xlim(0, fs/2)
        ax.set_ylim(-100, 5)
        ax.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
