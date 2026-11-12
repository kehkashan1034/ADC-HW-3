
"""
Part C: Rectangular and Raised-Cosine Pulse Shaping

This program compares rectangular and raised-cosine pulse shaping,
PSD, bandwidth, eye diagrams and the Nyquist zero-ISI condition.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

Path("plots").mkdir(exist_ok=True)


def rectangular_pulse(samples_per_symbol):
    """Generate a rectangular pulse lasting one symbol."""
    return np.ones(samples_per_symbol, dtype=float)


def raised_cosine(beta, span_symbols=8, samples_per_symbol=16):
    """Generate a finite-span, energy-normalized raised-cosine pulse."""

    if not 0 <= beta <= 1:
        raise ValueError("Roll-off beta must be between 0 and 1.")

    half_samples = span_symbols * samples_per_symbol // 2

    t = np.arange(-half_samples, half_samples + 1) / samples_per_symbol

    pulse = np.zeros_like(t, dtype=float)

    for i, x in enumerate(t):

        if np.isclose(x, 0.0):
            pulse[i] = 1.0

        elif beta > 0 and np.isclose(abs(x), 1 / (2 * beta)):
            pulse[i] = (np.pi / 4) * np.sinc(1 / (2 * beta))

        else:
            denominator = 1 - (2 * beta * x) ** 2

            pulse[i] = (
                np.sinc(x) * np.cos(np.pi * beta * x)
                / denominator
            )

    energy = np.sum(pulse ** 2)

    return pulse / np.sqrt(energy)


def upsample_symbols(symbols, samples_per_symbol):
    """Insert zeros between consecutive symbols."""

    result = np.zeros(len(symbols) * samples_per_symbol)

    result[::samples_per_symbol] = symbols

    return result


def shape_signal(symbols, pulse, samples_per_symbol):
    """Apply pulse shaping and compensate for filter group delay."""

    upsampled = upsample_symbols(symbols, samples_per_symbol)

    full_signal = np.convolve(upsampled, pulse, mode="full")

    group_delay = (len(pulse) - 1) // 2

    return full_signal[
        group_delay:group_delay + len(upsampled)
    ]


def eye_diagram(ax, signal, sps, title, traces=100):
    """Plot overlapping two-symbol waveform segments."""

    span = 2 * sps

    number = min(
        traces,
        max(0, (len(signal) - span) // sps)
    )

    time_axis = np.arange(span) / sps

    for k in range(number):

        start = k * sps

        ax.plot(
            time_axis,
            signal[start:start + span],
            alpha=0.18,
            linewidth=0.8
        )

    ax.set_title(title)
    ax.set_xlabel("Time (symbol intervals)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)


def estimate_occupied_bandwidth(freq, psd, fraction=0.99):
    """
    Estimate positive-frequency bandwidth containing 99%
    of the positive-frequency PSD area.
    """

    positive = freq >= 0

    f = freq[positive]
    p = np.maximum(psd[positive], 0)

    if len(f) < 2:
        return 0.0

    df = f[1] - f[0]

    cumulative_power = np.cumsum(p) * df

    if cumulative_power[-1] <= 0:
        return 0.0

    index = np.searchsorted(
        cumulative_power,
        fraction * cumulative_power[-1]
    )

    return float(f[min(index, len(f) - 1)])


def nyquist_sample_values(pulse, sps, symbol_offsets=4):
    """Evaluate pulse amplitudes at integer-symbol sampling instants."""

    center = (len(pulse) - 1) // 2

    offsets = np.arange(
        -symbol_offsets,
        symbol_offsets + 1
    )

    indices = center + offsets * sps

    valid = (
        (indices >= 0) &
        (indices < len(pulse))
    )

    return offsets[valid], pulse[indices[valid]]


def main():

    rng = np.random.default_rng(12)

    sps = 16
    symbol_period = 1.0
    fs = sps / symbol_period

    symbols = rng.choice([-1.0, 1.0], 3000)

    beta_values = [0.0, 0.25, 0.5, 1.0]

    # Generate pulse shapes.

    pulses = {
        "Rectangular": rectangular_pulse(sps)
    }

    for beta in beta_values:

        pulses[f"Raised Cosine beta={beta:g}"] = raised_cosine(
            beta,
            span_symbols=8,
            samples_per_symbol=sps
        )

    # Generate shaped signals.

    shaped_signals = {
        name: shape_signal(symbols, pulse, sps)
        for name, pulse in pulses.items()
    }


    # 1. PSD COMPARISON
    

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes = axes.flatten()

    bandwidths = []

    for ax, (name, signal) in zip(
        axes,
        shaped_signals.items()
    ):

        freq, psd = welch(
            signal,
            fs=fs,
            nperseg=4096,
            noverlap=2048,
            window="hann",
            detrend=False,
            scaling="density"
        )

        ax.semilogy(
            freq,
            np.maximum(psd, 1e-12)
        )

        ax.set_xlim(0, 1.5)
        ax.set_title(name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.grid(True)

        bandwidth = estimate_occupied_bandwidth(
            freq,
            psd,
            fraction=0.99
        )

        bandwidths.append((name, bandwidth))

    axes[-1].axis("off")

    fig.suptitle("Part C: PSD Comparison of Pulse Shapes")

    fig.tight_layout()

    fig.savefig(
        "plots/part_C_PSD.png",
        dpi=300
    )

    plt.show()

    
    # 2. THEORETICAL BANDWIDTH VS ROLL-OFF FACTOR
  

    betas = np.asarray(beta_values)

    theoretical_bw = (
        (1 + betas) / (2 * symbol_period)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        betas,
        theoretical_bw,
        "o-",
        linewidth=2
    )

    ax.set_xlabel("Roll-off factor (beta)")
    ax.set_ylabel("One-sided baseband bandwidth (Hz)")

    ax.set_title(
        "Raised-Cosine Bandwidth versus Roll-Off Factor"
    )

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        "plots/part_C_bandwidth.png",
        dpi=300
    )

    plt.show()

    
    # 3. EYE DIAGRAMS
  

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes = axes.flatten()

    for ax, (name, signal) in zip(
        axes,
        shaped_signals.items()
    ):

        eye_diagram(
            ax,
            signal[:300 * sps],
            sps,
            name,
            traces=100
        )

    axes[-1].axis("off")

    fig.suptitle(
        "Part C: Eye Diagrams for Different Pulse Shapes"
    )

    fig.tight_layout()

    fig.savefig(
        "plots/part_C_eye_diagrams.png",
        dpi=300
    )

    plt.show()

    
    # 4. NYQUIST ZERO-ISI CHECK
  

    print("\nNYQUIST ZERO-ISI CHECK")

    for beta in beta_values:

        name = f"Raised Cosine beta={beta:g}"

        offsets, values = nyquist_sample_values(
            pulses[name],
            sps,
            symbol_offsets=4
        )

        off_symbol = offsets != 0

        maximum_residual = np.max(
            np.abs(values[off_symbol])
        )

        print(
            f"Beta={beta:.2f}: "
            f"Maximum off-center pulse value = "
            f"{maximum_residual:.6e}"
        )

   
    # 5. DISPLAY BANDWIDTH RESULTS
    

    print("\nTHEORETICAL RAISED-COSINE BANDWIDTHS")

    for beta, bandwidth in zip(
        betas,
        theoretical_bw
    ):

        print(
            f"Beta={beta:.2f}: "
            f"{bandwidth:.4f} Hz"
        )

    print("\nESTIMATED 99% PSD BANDWIDTHS")

    for name, bandwidth in bandwidths:

        print(
            f"{name}: {bandwidth:.4f} Hz"
        )

    print(
        "\nNote: An ideal rectangular pulse has infinite "
        "frequency support. Its displayed bandwidth is a "
        "finite-record 99% PSD estimate."
    )

    print("\nPart C completed successfully.")

    print("All figures are saved in the plots folder.")


if __name__ == "__main__":
    main()
