
"""
Part D: Multipath Channel and Equalization

This program investigates multipath-induced ISI and compares
Zero-Forcing (ZF) and Decision Feedback Equalization (DFE).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("plots").mkdir(exist_ok=True)


def bpsk_modulate(bits):
    """Convert binary bits into BPSK symbols -1 and +1."""

    return 2.0 * np.asarray(bits, dtype=float) - 1.0


def make_oversampled_waveform(symbols, sps):
    """Generate an oversampled rectangular NRZ waveform."""

    return np.repeat(symbols, sps)


def make_oversampled_channel(symbol_taps, sps):
    """Place channel taps at symbol-spaced sample positions."""

    symbol_taps = np.asarray(symbol_taps, dtype=float)

    channel = np.zeros(
        (len(symbol_taps) - 1) * sps + 1
    )

    channel[::sps] = symbol_taps

    return channel


def multipath_channel(waveform, channel_taps, noise_std, rng):
    """Apply multipath channel and add Gaussian noise."""

    received = np.convolve(
        waveform,
        channel_taps,
        mode="full"
    )

    received = received[:len(waveform)]

    received += rng.normal(
        0.0,
        noise_std,
        size=len(received)
    )

    return received


def design_zf_equalizer(channel_taps, n_taps, decision_delay):
    """Design a finite-length Zero-Forcing FIR equalizer."""

    h = np.asarray(channel_taps, dtype=float)

    rows = len(h) + n_taps - 1

    matrix = np.zeros(
        (rows, n_taps)
    )

    for col in range(n_taps):

        matrix[
            col:col + len(h),
            col
        ] = h

    if not 0 <= decision_delay < rows:
        raise ValueError("Invalid decision delay.")

    target = np.zeros(rows)

    target[decision_delay] = 1.0

    equalizer, *_ = np.linalg.lstsq(
        matrix,
        target,
        rcond=None
    )

    return equalizer


def apply_fir_full(signal, filter_taps):
    """Apply an FIR filter using full convolution."""

    return np.convolve(
        signal,
        filter_taps,
        mode="full"
    )


def decision_feedback_equalizer(received, channel_taps):
    """
    Perform symbol-by-symbol DFE.

    Previous detected symbols are used to cancel
    post-cursor intersymbol interference.
    """

    h = np.asarray(channel_taps, dtype=float)

    if len(h) == 0 or abs(h[0]) < 1e-12:
        raise ValueError("First channel tap must be nonzero.")

    decisions = np.zeros(
        len(received),
        dtype=float
    )

    for k in range(len(received)):

        estimated_isi = 0.0

        for j in range(1, len(h)):

            previous_index = k - j

            if previous_index >= 0:

                estimated_isi += (
                    h[j] * decisions[previous_index]
                )

        estimate = (
            received[k] - estimated_isi
        ) / h[0]

        decisions[k] = (
            1.0 if estimate >= 0 else -1.0
        )

    return decisions


def calculate_ber(reference, detected, start=0):
    """Calculate BER while excluding initial transient symbols."""

    n = min(
        len(reference),
        len(detected)
    )

    if n <= start:
        raise ValueError("Not enough samples for BER calculation.")

    return np.mean(
        reference[start:n] != detected[start:n]
    )


def eye_diagram(ax, signal, sps, title, traces=100):
    """Plot eye diagrams using an oversampled waveform."""

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


def main():

    rng = np.random.default_rng(23)


    # 1. SIMULATION PARAMETERS
   

    nbits = 30000

    sps = 8

    channel_symbol_taps = np.array([
        1.0,
        0.55,
        0.25
    ])

    ebn0_db = 14.0

  
    # 2. BPSK TRANSMISSION
 
    bits = rng.integers(
        0,
        2,
        nbits
    )

    symbols = bpsk_modulate(bits)

    ebn0_linear = 10 ** (ebn0_db / 10)

    noise_std = np.sqrt(
        1 / (2 * ebn0_linear)
    )

    tx_waveform = make_oversampled_waveform(
        symbols,
        sps
    )

   
    # 3. MULTIPATH CHANNEL
    

    channel_samples = make_oversampled_channel(
        channel_symbol_taps,
        sps
    )

    received_waveform = multipath_channel(
        tx_waveform,
        channel_samples,
        noise_std,
        rng
    )

    # Sample at the centre of each symbol interval.

    sample_indices = (
        np.arange(nbits) * sps + sps // 2
    )

    received_samples = received_waveform[
        sample_indices
    ]


    # 4. DETECTION BEFORE EQUALIZATION


    raw_decisions = np.where(
        received_samples >= 0,
        1.0,
        -1.0
    )

    
    # 5. ZERO-FORCING EQUALIZATION
    

    zf_delay_symbols = 4

    zf_delay_samples = zf_delay_symbols * sps

    zf_filter_length = 9 * sps + 1

    zf_taps = design_zf_equalizer(
        channel_samples,
        n_taps=zf_filter_length,
        decision_delay=zf_delay_samples
    )

    zf_full_output = apply_fir_full(
        received_waveform,
        zf_taps
    )

    # Compensate for equalizer decision delay.

    zf_aligned_waveform = zf_full_output[
        zf_delay_samples:
        zf_delay_samples + len(tx_waveform)
    ]

    zf_symbol_samples = zf_aligned_waveform[
        sample_indices
    ]

    zf_decisions = np.where(
        zf_symbol_samples >= 0,
        1.0,
        -1.0
    )

    
    # 6. DECISION FEEDBACK EQUALIZATION
  

    dfe_decisions = decision_feedback_equalizer(
        received_samples,
        channel_symbol_taps
    )

   
    # 7. BER CALCULATION
    

    transient_symbols = 20

    raw_ber = calculate_ber(
        symbols,
        raw_decisions,
        start=transient_symbols
    )

    zf_ber = calculate_ber(
        symbols,
        zf_decisions,
        start=transient_symbols
    )

    dfe_ber = calculate_ber(
        symbols,
        dfe_decisions,
        start=transient_symbols
    )

    print("\nMULTIPATH CHANNEL PARAMETERS")

    print(
        f"Channel taps: {channel_symbol_taps}"
    )

    print(
        f"Samples per symbol: {sps}"
    )

    print(
        f"Eb/N0: {ebn0_db:.1f} dB"
    )

    print("\nBER RESULTS")

    print(
        f"BER before equalization: {raw_ber:.6g}"
    )

    print(
        f"BER after ZF equalization: {zf_ber:.6g}"
    )

    print(
        f"BER after DFE: {dfe_ber:.6g}"
    )

    print(
        f"\nZF equalizer length: {len(zf_taps)} samples"
    )

    print(
        f"ZF decision delay: {zf_delay_symbols} symbols"
    )


    # 8. RECEIVED AND EQUALIZED SIGNAL COMPARISON
    

    count_symbols = 80

    count_samples = count_symbols * sps

    time_axis = (
        np.arange(count_samples) / sps
    )

    dfe_waveform = np.repeat(
        dfe_decisions[:count_symbols],
        sps
    )

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        time_axis,
        tx_waveform[:count_samples],
        label="Transmitted NRZ waveform",
        linewidth=1.5
    )

    ax.plot(
        time_axis,
        received_waveform[:count_samples],
        label="Received after multipath",
        alpha=0.75
    )

    ax.plot(
        time_axis,
        zf_aligned_waveform[:count_samples],
        label="After ZF equalization",
        alpha=0.85
    )

    ax.plot(
        time_axis,
        dfe_waveform,
        label="DFE symbol decisions (held)",
        alpha=0.8
    )

    ax.set_title(
        "Part D: Transmitted, Received and Equalized Signals"
    )

    ax.set_xlabel("Time (symbol intervals)")
    ax.set_ylabel("Amplitude")

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        "plots/part_D_received_signals.png",
        dpi=300
    )

    plt.show()

    
    # 9. EYE DIAGRAMS
   

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.5)
    )

    eye_diagram(
        axes[0],
        received_waveform,
        sps,
        "Before Equalization (Multipath)"
    )

    eye_diagram(
        axes[1],
        zf_aligned_waveform,
        sps,
        "After ZF Equalization"
    )

    # DFE provides symbol decisions, not an analog waveform.
    # Display these decisions as a held waveform.

    eye_diagram(
        axes[2],
        np.repeat(dfe_decisions, sps),
        sps,
        "After DFE (Held Decisions)"
    )

    fig.suptitle(
        "Part D: Eye Diagrams Before and After Equalization"
    )

    fig.tight_layout()

    fig.savefig(
        "plots/part_D_eye_diagrams.png",
        dpi=300
    )

    plt.show()

    
    # 10. BER COMPARISON
    

    labels = [
        "Before EQ",
        "ZF",
        "DFE"
    ]

    values = [
        raw_ber,
        zf_ber,
        dfe_ber
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plotting floor allows zero-error results on log scale.

    plot_values = [
        max(value, 0.5 / nbits)
        for value in values
    ]

    ax.bar(
        labels,
        plot_values
    )

    ax.set_yscale("log")

    ax.set_ylabel("Bit Error Rate (BER)")

    ax.set_title(
        f"BER Before and After Equalization "
        f"(Eb/N0 = {ebn0_db:.1f} dB)"
    )

    ax.grid(
        True,
        axis="y",
        which="both"
    )

    fig.tight_layout()

    fig.savefig(
        "plots/part_D_BER.png",
        dpi=300
    )

    plt.show()

    print("\nPart D completed successfully.")

    print(
        "All figures are saved in the plots folder."
    )

    print(
        "Note: The DFE eye plot displays hard symbol "
        "decisions held over each symbol interval."
    )


if __name__ == "__main__":
    main()
