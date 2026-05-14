from pathlib import Path
import warnings
import numpy as np
import obspy
import matplotlib.pyplot as plt
from scipy.signal import periodogram, detrend
from scipy.signal.windows import dpss

"""Compute the MERMAID power spectral density (PSD) for a given set of seismograms, and save plots."""

warnings.filterwarnings(
    "ignore",
    message=r"Sample spacing read from SAC file.*was rounded.*microsecond precision",
    category=UserWarning,
    module=r"obspy\.io\.sac\.util",
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
sac_dir = Path("./Mermaid_Data_Joel")
out_dir = Path("./plots/Noise/Mermaid_Noise_Spectra")
out_dir.mkdir(parents=True, exist_ok=True)

noise_seconds = 80

def noise_spectrum_from_sac(sac_file, save_plot=True):
    st = obspy.read(str(sac_file))
    tr = st[0]

    dt = tr.stats.delta
    fs = 1.0 / dt

    n_noise = int(round(noise_seconds * fs))

    if tr.stats.npts < n_noise:
        raise ValueError(f"{sac_file} is shorter than {noise_seconds} s")

    # First noise_seconds are assumed to be noise.
    x = tr.data[:n_noise].astype(float)

    # Basic preprocessing for spectrum.
    x = detrend(x, type="linear")
    x = x - np.mean(x)

    # Paper-like single tapered spectrum.
    taper = dpss(len(x), NW=4, Kmax=4, sym=False)[0]

    freq, psd = periodogram(
        x,
        fs=fs,
        window=taper,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )

    # Avoid log10(0).
    psd_db = 10.0 * np.log10(np.maximum(psd, np.finfo(float).tiny))

    if save_plot:
        fig, ax = plt.subplots(figsize=(8, 4.8))

        # Main PSD curve
        ax.semilogx(
            freq,
            psd_db,
            color="k",
            linewidth=1.4,
        )

        ax.set_xlim(0.01, min(fs / 2, 10.0))

        # Set y limits robustly so one spike does not ruin the plot
        good = np.isfinite(psd_db) & (freq >= 0.01) & (freq <= min(fs / 2, 10.0))
        if np.any(good):
            ylo, yhi = np.nanpercentile(psd_db[good], [0, 100])
            pad = 0.12 * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + pad)

        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel(r"$10 \log_{10}$ PSD " + r"counts$^2$/Hz", fontsize=12)

        ax.set_title(
            f"{sac_file.name}\n"
            f"Noise spectrum from first {noise_seconds:.0f} s",
            fontsize=12,
            pad=10,
        )

        ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
        ax.grid(True, which="minor", linewidth=0.5, alpha=0.18)

        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", labelsize=9)

        # Add metadata text box
        text = (
            f"Start: {tr.stats.starttime.datetime}\n"
            f"$\\Delta t$ = {dt:.4f} s"
        )

        ax.text(
            0.02,
            0.04,
            text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.9,
            ),
        )

        fig.tight_layout()

        plot_file = out_dir / f"{sac_file.name}_noise_spectrum.pdf"
        fig.savefig(plot_file, dpi=250, bbox_inches="tight")
        plt.close(fig)

    return {
        "freq": freq,
        "psd": psd,
        "psd_db": psd_db,
        "starttime": tr.stats.starttime.datetime,
        "dt": dt,
        "fs": fs,
        "sac_file": str(sac_file),
    }


# ------------------------------------------------------------
# Run all SAC files
# ------------------------------------------------------------
sac_files = sorted(sac_dir.glob("*"))

for sac_file in sac_files[:10]:
    try:
        result = noise_spectrum_from_sac(sac_file, save_plot=True)

        print(f"Saved plot for {sac_file.name}")

    except Exception as e:
        print(f"Skipped {sac_file}: {e}")