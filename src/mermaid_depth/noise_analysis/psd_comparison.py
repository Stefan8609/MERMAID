from pathlib import Path
from datetime import datetime, timedelta
import warnings

import numpy as np
import xarray as xr
import obspy
import matplotlib.pyplot as plt

from scipy.signal import periodogram, detrend
from scipy.signal.windows import dpss


warnings.filterwarnings(
    "ignore",
    message=r"Sample spacing read from SAC file.*was rounded.*microsecond precision",
    category=UserWarning,
    module=r"obspy\.io\.sac\.util",
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
N_MERMAIDS_TO_PLOT = 10  # Set to None to plot all

mermaid_file = Path("./saved_data/mermaid_loc_time.npz")
sac_dir = Path("./Mermaid_Data_Joel")
ww3_dir = Path("./WaveWatch_Data/raw")

out_dir = Path("./plots/Noise/Mermaid_WaveWatch_Comparison")
out_dir.mkdir(parents=True, exist_ok=True)

noise_seconds = 90.0

# MERMAID sensitivity: counts per Pa.
# Sign does not matter for PSD.
COUNTS_PER_PA = 1.494e5

fmin_plot = 1e-2
fmax_plot = 2.0


def lon_to_180(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def parse_time(t):
    """
    Handles normal ISO strings and TOMOCAT-style strings.

    Normal:
        2026-03-30T08:54:50.418058

    TOMOCAT-style:
        2026-03-30T08:54:500.418058
    """
    if isinstance(t, bytes):
        t = t.decode()

    t = str(t)

    try:
        return np.datetime64(datetime.fromisoformat(t))
    except ValueError:
        date_part, clock_part = t.split("T")
        hh, mm, sec_plus_b = clock_part.split(":")

        sec = sec_plus_b[:2]
        b_string = sec_plus_b[2:]

        b_offset = float(b_string) if b_string else 0.0

        ref_time = datetime.strptime(
            f"{date_part}T{hh}:{mm}:{sec}",
            "%Y-%m-%dT%H:%M:%S",
        )

        return np.datetime64(ref_time + timedelta(seconds=b_offset))


def safe_name(x):
    if isinstance(x, bytes):
        x = x.decode()

    x = Path(str(x)).name

    for bad in ["/", "\\", ":", " ", "\t"]:
        x = x.replace(bad, "_")

    return x


def get_sac_path(name):
    if isinstance(name, bytes):
        name = name.decode()

    name = str(name)

    p = Path(name)

    if p.exists():
        return p

    p = sac_dir / Path(name).name

    return p


def mermaid_noise_psd(sac_file):
    st = obspy.read(str(sac_file))
    tr = st[0]

    dt = tr.stats.delta
    fs = 1.0 / dt

    n_noise = int(round(noise_seconds * fs))

    if tr.stats.npts < n_noise:
        raise ValueError(f"{sac_file} is shorter than {noise_seconds} s")

    # First noise_seconds are assumed to be noise.
    x = tr.data[:n_noise].astype(float)

    # Detrend and demean.
    x = detrend(x, type="linear")
    x = x - np.mean(x)

    # Convert counts to pressure in Pa before PSD.
    x = x / COUNTS_PER_PA

    taper = dpss(len(x), NW=4, Kmax=1, sym=False)[0]

    freq, psd = periodogram(
        x,
        fs=fs,
        window=taper,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )

    psd_db = 10.0 * np.log10(np.maximum(psd, np.finfo(float).tiny))

    return freq, psd_db, tr.stats.starttime.datetime, fs


# ------------------------------------------------------------
# Load MERMAID metadata
# ------------------------------------------------------------
data = np.load(mermaid_file, allow_pickle=True)

stla = data["stla"].astype(float)
stlo = data["stlo"].astype(float)
filenames = data["filename"]
seismogram_time = data["seismogram_time"]

times = np.array([parse_time(t) for t in seismogram_time])

if N_MERMAIDS_TO_PLOT is None:
    n_plot = len(times)
else:
    n_plot = min(N_MERMAIDS_TO_PLOT, len(times))


# ------------------------------------------------------------
# Make comparison plots
# ------------------------------------------------------------
for i in range(n_plot):
    t = times[i]
    t_str = np.datetime_as_string(t, unit="s")

    year = int(t_str[:4])
    month = int(t_str[5:7])
    ym = f"{year}{month:02d}"

    p2l_file = ww3_dir / f"LOPS_WW3-GLOB-30M_{ym}_p2l.nc"
    sac_file = get_sac_path(filenames[i])

    if not sac_file.exists():
        print(f"[{i}] Missing SAC file: {sac_file}")
        continue

    if not p2l_file.exists():
        print(f"[{i}] Missing WW3 file: {p2l_file}")
        continue

    print(f"[{i}] plotting {safe_name(filenames[i])}")
    print(f"    SAC: {sac_file}")
    print(f"    WW3: {p2l_file.name}")

    try:
        # ------------------------------------------------------------
        # MERMAID PSD
        # ------------------------------------------------------------
        freq_mermaid, mermaid_psd_db, sac_starttime, fs = mermaid_noise_psd(sac_file)

        # ------------------------------------------------------------
        # WW3 P2L nearest time/lat/lon
        # ------------------------------------------------------------
        with xr.open_dataset(p2l_file) as ds:
            ww3_lons = ds["longitude"].values

            if ww3_lons.max() > 180:
                lon_query = stlo[i] % 360.0
            else:
                lon_query = lon_to_180(stlo[i])

            p = ds["p2l"].sel(
                time=t,
                latitude=stla[i],
                longitude=lon_query,
                method="nearest",
            )

            selected_time = np.datetime_as_string(p["time"].values, unit="s")
            selected_lat = float(p["latitude"].values)
            selected_lon = float(p["longitude"].values)

            if selected_lon > 180:
                selected_lon = lon_to_180(selected_lon)

            # Native WW3 frequency coordinate.
            freq_ww3 = p["f"].values

            # IFREMER P2L files store p2l as log10(Pa^2/Hz).
            # Therefore 10*p2l gives 10 log10(Pa^2/Hz).
            p2l_db = 10.0 * p.values

        # ------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.5, 5.0))

        ax.semilogx(
            freq_mermaid,
            mermaid_psd_db,
            color="k",
            linewidth=1.4,
            label=f"MERMAID PSD, first {noise_seconds:.0f} s",
        )

        ax.semilogx(
            freq_ww3,
            p2l_db,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label="WW3 P2L",
        )

        ax.set_xlim(fmin_plot, fmax_plot)

        good_mermaid = (
            np.isfinite(mermaid_psd_db)
            & (freq_mermaid >= fmin_plot)
            & (freq_mermaid <= fmax_plot)
        )

        good_ww3 = (
            np.isfinite(p2l_db)
            & (freq_ww3 >= fmin_plot)
            & (freq_ww3 <= fmax_plot)
        )

        y_parts = []
        if np.any(good_mermaid):
            y_parts.append(mermaid_psd_db[good_mermaid])
        if np.any(good_ww3):
            y_parts.append(p2l_db[good_ww3])

        if y_parts:
            y_all = np.concatenate(y_parts)
            ylo, yhi = np.nanpercentile(y_all, [2, 98])
            pad = 0.15 * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + pad)

        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.18)

        ax.set_xlabel("Native frequency (Hz)")
        ax.set_ylabel(r"$10 \log_{10}$ PSD / P2L (Pa$^2$/Hz)")

        ax.set_title(
            f"MERMAID noise PSD vs. matched WW3 P2L\n"
            f"{safe_name(filenames[i])}\n"
            f"MERMAID: {t_str}, lat={stla[i]:.3f}, lon={stlo[i]:.3f}\n"
            f"WW3 nearest: {selected_time}, lat={selected_lat:.3f}, lon={selected_lon:.3f}"
        )

        ax.legend(frameon=True, fontsize=9)

        fig.tight_layout()

        plot_file = out_dir / f"{i:05d}_{safe_name(filenames[i])}_mermaid_vs_ww3_native.png"
        fig.savefig(plot_file, dpi=250, bbox_inches="tight")
        plt.close(fig)

        print(f"    saved {plot_file}")

    except Exception as e:
        print(f"[{i}] skipped because of error: {e}")

print("\nDone.")