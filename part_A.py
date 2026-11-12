
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path


#creating the plots folder if it does not exist
Path("plots").mkdir(exist_ok=True)


def generate_line_codes(bits, A=1, samples_per_bit=100):
    """
    Generate four baseband line-coding schemes:
    Unipolar NRZ, Unipolar RZ, Polar NRZ, and Polar RZ.
    """

    if samples_per_bit <= 0 or samples_per_bit % 2 != 0:
        raise ValueError("samples_per_bit must be a positive even number.")

    # Unipolar NRZ: 1 -> A, 0 -> 0
    unipolar_nrz = np.repeat(A * bits, samples_per_bit)

    # Unipolar RZ: 1 -> A for half a bit, then 0
    unipolar_rz = unipolar_nrz.copy()
    unipolar_rz = unipolar_rz.reshape(-1, samples_per_bit)
    unipolar_rz[:, samples_per_bit // 2:] = 0
    unipolar_rz = unipolar_rz.flatten()

    # Polar NRZ: 1 -> +A, 0 -> -A
    polar_levels = np.where(bits == 1, A, -A)
    polar_nrz = np.repeat(polar_levels, samples_per_bit)

    # Polar RZ: +A or -A for half a bit, then 0
    polar_rz = np.repeat(polar_levels, samples_per_bit)
    polar_rz = polar_rz.reshape(-1, samples_per_bit)
    polar_rz[:, samples_per_bit // 2:] = 0
    polar_rz = polar_rz.flatten()

    return {
        "Unipolar NRZ": unipolar_nrz,
        "Unipolar RZ": unipolar_rz,
        "Polar NRZ": polar_nrz,
        "Polar RZ": polar_rz
    }


def plot_line_codes(bits, signals, samples_per_bit=100):
    """
    Plot the time-domain waveforms of all four line codes.
    """

    time = np.arange(len(bits) * samples_per_bit) / samples_per_bit

    fig, axes = plt.subplots(4, 1, figsize=(12, 9))

    for ax, (name, signal) in zip(axes, signals.items()):
        ax.step(time, signal, where="post")
        ax.set_title(name)
        ax.set_xlabel("Time (bit intervals)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, len(bits))
        ax.grid(True)

    fig.suptitle("Time-Domain Representation of Line Codes")
    plt.tight_layout()
    plt.savefig("plots/part_A_line_codes.png", dpi=300)
    plt.show()


def plot_psd(signals, fs):
    """
    Estimate and plot the Power Spectral Density (PSD)
    using Welch's method.
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (name, signal) in zip(axes, signals.items()):

        frequencies, psd = welch(
            signal,
            fs=fs,
            window="hann",
            nperseg=1024,
            noverlap=512,
            detrend=False,
            scaling="density"
        )

        ax.semilogy(frequencies, np.maximum(psd, 1e-12))
        ax.set_title(name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (Amplitude²/Hz)")
        ax.grid(True)

    fig.suptitle("Power Spectral Density of Line-Coding Schemes")
    plt.tight_layout()
    plt.savefig("plots/part_A_PSD.png", dpi=300)
    plt.show()


def main():

    np.random.seed(42)

    A = 1
    samples_per_bit = 100

    # Generate a short sequence for clear time-domain plots
    time_bits = np.random.randint(0, 2, 10)

    time_signals = generate_line_codes(
        time_bits, A, samples_per_bit
    )

    print("Time-domain input bits:")
    print(time_bits)

    plot_line_codes(
        time_bits, time_signals, samples_per_bit
    )

    #generating a longer sequence for more reliable PSD estimates
    psd_bits = np.random.randint(0, 2, 5000)

    psd_signals = generate_line_codes(
        psd_bits, A, samples_per_bit
    )

    # assuming bit duration Tb = 1 second
    # sampling frequency = samples per bit / Tb
    fs = samples_per_bit

    print("\nGenerating PSD plots...")
    print("Number of bits used for PSD:", len(psd_bits))
    print("Sampling frequency:", fs, "Hz")

    plot_psd(psd_signals, fs)

    print("\nPart A completed successfully!")
    print("Plots saved in the 'plots' folder.")


if __name__ == "__main__":
    main()
