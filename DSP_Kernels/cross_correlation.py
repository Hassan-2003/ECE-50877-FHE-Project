import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt

def matched_filter(received_signal, reference_signal):
    """
    Apply a matched filter to a received signal.

    Parameters
    ----------
    received_signal : array_like
        Received signal.
    reference_signal : array_like
        Known transmitted/reference signal.

    Returns
    -------
    output : ndarray
        Matched filter output.
    """
    reference_signal = np.asarray(reference_signal)

    coefficients = np.conj(reference_signal[::-1])

    return lfilter(coefficients, 1.0, received_signal)


def main():
    # Generate reference signal
    reference_length = 128
    reference = np.random.randn(reference_length)

    # Generate received signal with noise
    delay = 200
    noise = 0.5 * np.random.randn(600)

    received = noise.copy()
    received[delay:delay + reference_length] += reference

    # Apply matched filter
    output = matched_filter(received, reference)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(reference)
    axes[0].set_title("Reference Signal")

    axes[1].plot(received)
    axes[1].set_title("Received Signal")

    axes[2].plot(np.abs(output))
    axes[2].set_title("Matched Filter Output")

    for ax in axes:
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
