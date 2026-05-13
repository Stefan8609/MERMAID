from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Global settings
# ------------------------------------------------------------
N_MERMAIDS_TO_PLOT = 10   # Set to None to plot all

mermaid_file = Path("./saved_data/mermaid_loc_time.npz")
ww3_dir = Path("./WaveWatch_Data/raw")
out_dir = Path("./plots/Noise/WaveWatch_P2L_Spectra")
out_dir.mkdir(parents=True, exist_ok=True)


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
# Plot matched P2L spectrum for each MERMAID
# ------------------------------------------------------------
for i in range(n_plot):
    t = times[i]
    t_str = np.datetime_as_string(t, unit="s")

    year = int(t_str[:4])
    month = int(t_str[5:7])
    ym = f"{year}{month:02d}"

    p2l_file = ww3_dir / f"LOPS_WW3-GLOB-30M_{ym}_p2l.nc"

    if not p2l_file.exists():
        print(f"[{i}] Missing WW3 file: {p2l_file}")
        continue

    print(f"[{i}] plotting {safe_name(filenames[i])} using {p2l_file.name}")

    with xr.open_dataset(p2l_file) as ds:
        # Match longitude convention of WW3 file.
        ww3_lons = ds["longitude"].values

        if ww3_lons.max() > 180:
            lon_query = stlo[i] % 360.0
        else:
            lon_query = lon_to_180(stlo[i])

        # Nearest WW3 time, latitude, longitude.
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

        freq_ww3 = p["f"].values

        # Most IFREMER p2l files store log10(Pa^2/Hz).
        # Plot as 10 log10(Pa^2/Hz), matching the paper style.
        p2l_db = 10.0 * p.values

    # ------------------------------------------------------------
    # Make plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.semilogx(
        freq_ww3,
        p2l_db,
        color="k",
        linewidth=1.5,
    )

    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.18)

    ax.set_xlabel("Equivalent MERMAID acoustic frequency, 2f (Hz)")
    ax.set_ylabel(r"WW3 P2L, $10 \log_{10}(\mathrm{Pa}^2/\mathrm{Hz})$")

    ax.set_title(
        f"Matched WW3 P2L spectrum\n"
        f"{safe_name(filenames[i])}\n"
        f"MERMAID: {t_str}, lat={stla[i]:.3f}, lon={stlo[i]:.3f}\n"
        f"WW3 nearest: {selected_time}, lat={selected_lat:.3f}, lon={selected_lon:.3f}"
    )

    ax.set_xlim(freq_ww3.min(), freq_ww3.max())

    fig.tight_layout()

    plot_file = out_dir / f"{i:05d}_{safe_name(filenames[i])}_p2l.pdf"
    fig.savefig(plot_file, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"    saved {plot_file}")

print("\nDone.")