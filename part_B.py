"""
Part B: AWGN Channel, Matched Filtering, ML Detection, and BER


The four line-code formats use a one-bit observation interval. For each bit,
the receiver compares the noisy observation with the two possible noiseless
bit waveforms. This is the ML rule for AWGN with equal bit probabilities.
The matched-filter statistic is the correlation with the difference between
the two bit templates; it gives the same decision as the ML distance test.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("plots").mkdir(exist_ok=True)


def bit_templates(code_name, A=1.0, samples_per_bit=100):
    """Return the noiseless sample templates for bit 0 and bit 1."""
    half = samples_per_bit // 2
    if code_name == "Unipolar NRZ":
        return np.zeros(samples_per_bit), np.full(samples_per_bit, A)
    if code_name == "Unipolar RZ":
        return np.zeros(samples_per_bit), np.r_[np.full(half, A), np.zeros(samples_per_bit-half)]
    if code_name == "Polar NRZ":
        return np.full(samples_per_bit, -A), np.full(samples_per_bit, A)
    if code_name == "Polar RZ":
        return np.r_[np.full(half, -A), np.zeros(samples_per_bit-half)], np.r_[np.full(half, A), np.zeros(samples_per_bit-half)]
    raise ValueError(f"Unknown line code: {code_name}")


def transmit_bits(bits, template0, template1):
    """Map bits to their corresponding line-code waveforms."""
    return np.concatenate([template1 if b else template0 for b in bits])


def add_awgn(signal, noise_variance, rng):
    """Add real, zero-mean, independent Gaussian noise to a sampled signal."""
    return signal + rng.normal(0.0, np.sqrt(noise_variance), size=signal.shape)


def detect_ml(received, template0, template1):
    """ML detector: choose the bit template with the smaller squared distance."""
    s = len(template0)
    blocks = received[:(len(received)//s)*s].reshape(-1, s)
    d0 = np.sum((blocks-template0)**2, axis=1)
    d1 = np.sum((blocks-template1)**2, axis=1)
    return (d1 < d0).astype(int)


def detect_matched_filter(received, template0, template1):
    """Matched-filter detector using correlation with template1-template0."""
    s = len(template0)
    blocks = received[:(len(received)//s)*s].reshape(-1, s)
    difference = template1-template0
    threshold = 0.5 * (np.sum(template1**2)-np.sum(template0**2))
    statistic = blocks @ difference
    return (statistic > threshold).astype(int)


def estimate_ber(bits, detected):
    """Calculate the fraction of incorrectly detected bits."""
    n = min(len(bits), len(detected))
    return np.mean(bits[:n] != detected[:n])



def eye_diagram(ax, signal, samples_per_bit, title, traces=60):
    """Display synchronized two-bit intervals with a visible eye opening."""

    span = 2 * samples_per_bit

    count = min(
        traces,
        (len(signal) - span) // samples_per_bit
    )

    for k in range(count):
        start = k * samples_per_bit
        segment = signal[start:start + span]

        ax.plot(
            np.arange(span) / samples_per_bit,
            segment,
            linewidth=0.8,
            alpha=0.22
        )

    ax.set_title(title)
    ax.set_xlabel("Time (bit intervals)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, 2)
    ax.grid(True)



def ber_sweep(code_name, A, ebn0_db, nbits=100_000, samples_per_bit=20, seed=7):
    """Simulate BER across Eb/N0 values using both ML and matched-filter detectors."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, nbits)
    t0, t1 = bit_templates(code_name, A, samples_per_bit)
    clean = transmit_bits(bits, t0, t1)
    # Average bit energy in the sampled-waveform convention.
    eb = np.mean([np.sum(t0**2), np.sum(t1**2)])
    ber_ml, ber_mf = [], []
    for db in ebn0_db:
        gamma = 10**(db/10)
        noise_var = eb/(2*gamma)
        noisy = add_awgn(clean, noise_var, rng)
        ber_ml.append(estimate_ber(bits, detect_ml(noisy, t0, t1)))
        ber_mf.append(estimate_ber(bits, detect_matched_filter(noisy, t0, t1)))
    return np.asarray(ber_ml), np.asarray(ber_mf)


def main():
    rng = np.random.default_rng(42)
    A = 1.0
    sps = 100
    nbits = 5000
    bits = rng.integers(0, 2, nbits)
    names = ["Unipolar NRZ", "Unipolar RZ", "Polar NRZ", "Polar RZ"]

    # Eye diagrams at a representative Eb/N0.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name in zip(axes.flat, names):
        t0, t1 = bit_templates(name, A, sps)
        clean = transmit_bits(bits, t0, t1)
        eb = np.mean([np.sum(t0**2), np.sum(t1**2)])
        noisy = add_awgn(clean, eb/(2*10**(12/10)), rng)
        eye_diagram(ax, noisy, sps, name)
    fig.suptitle("Part B: Eye Diagrams in AWGN (Eb/N0 = 12 dB)")
    fig.tight_layout()
    fig.savefig("plots/part_B_eye_diagrams.png", dpi=300)
    plt.show()

    
    ebn0_db = np.arange(0, 13, 2)

    ber_results = {}

    for name in names:

        ml, mf = ber_sweep(
            name,
            A,
            ebn0_db,
            samples_per_bit=20
        )

        ber_results[name] = (ml, mf)

    
    # FOUR INDIVIDUAL BER GRAPHS
    

    for name in names:

        ml, mf = ber_results[name]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.semilogy(
            ebn0_db,
            np.maximum(ml, 0.5/100000),
            "o-",
            label="ML Detection"
        )

        ax.semilogy(
            ebn0_db,
            np.maximum(mf, 0.5/100000),
            "s--",
            label="Matched Filter"
        )

        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("BER")
        ax.set_title(f"BER vs Eb/N0 - {name}")

        ax.grid(True, which="both")
        ax.legend()

        plt.tight_layout()

        filename = name.lower().replace(" ", "_")

        plt.savefig(
            f"plots/part_B_BER_{filename}.png",
            dpi=300
        )

        plt.show()

   
    # ONE COMBINED BER GRAPH
  

    fig, ax = plt.subplots(figsize=(11, 7))

    for name in names:

        ml, mf = ber_results[name]

        ax.semilogy(
            ebn0_db,
            np.maximum(ml, 0.5/100000),
            "o-",
            label=f"{name} (ML)"
        )

        ax.semilogy(
            ebn0_db,
            np.maximum(mf, 0.5/100000),
            "--",
            label=f"{name} (Matched Filter)"
        )

    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit Error Rate")

    ax.set_title("Combined BER Comparison of All Line Codes")

    ax.grid(True, which="both")
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()

    plt.savefig(
        "plots/part_B_BER_combined.png",
        dpi=300
    )

    plt.show()


    # Amplitude experiment at fixed absolute noise variance.
    test_bits = rng.integers(0, 2, 100_000)
    name = "Polar NRZ"
    t0a, t1a = bit_templates(name, 1.0, 20)
    eb_a = np.mean([np.sum(t0a**2), np.sum(t1a**2)])
    fixed_noise_var = eb_a/(2*10**(8/10))
    amp_results = []
    for amp in (1.0, 2.0):
        t0, t1 = bit_templates(name, amp, 20)
        clean = transmit_bits(test_bits, t0, t1)
        noisy = add_awgn(clean, fixed_noise_var, rng)
        amp_results.append(estimate_ber(test_bits, detect_ml(noisy, t0, t1)))
    print("Fixed-noise amplitude experiment")
    print(f"BER at A=1: {amp_results[0]:.6g}")
    print(f"BER at A=2: {amp_results[1]:.6g}")
    print("Note: if noise is instead rescaled to keep Eb/N0 constant, the BER curves overlap.")

    # Amplitude comparison graph
    plt.figure(figsize=(7, 5))

    plt.bar(
        ["A = 1", "A = 2"],
        amp_results
    )

    plt.ylabel("Bit Error Rate")
    plt.title("Effect of Transmit Amplitude on BER (Polar NRZ)")
    plt.yscale("log")
    plt.grid(True, axis="y", which="both")

    plt.tight_layout()
    plt.savefig(
        "plots/part_B_amplitude_comparison.png",
        dpi=300
    )
    plt.show()

    print("\nPart B completed. Plots saved in the 'plots' folder.")


if __name__ == "__main__":
    main()
