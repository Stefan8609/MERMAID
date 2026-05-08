import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mermaid_depth.depth_determination.source_time import (
    gaussian_sinusoid,
    source_time_spike_train,
)
from mermaid_depth.misc.read_tomocat1 import get_mermaid_data

def correlate_first_arrival(
    waveform,
    time,
    first_arrival_time=None,
    template_before=1.0,
    template_after=3.0,
    min_corr=0.3,
    min_separation=1.0,
    plot=True,
    amp_arrivals=None,
    t_arrivals=None,
):

    waveform = np.asarray(waveform, dtype=float)
    time = np.asarray(time, dtype=float)

    dt = time[1] - time[0]
    n = len(waveform)

    # ------------------------------------------------------------
    # Estimate first arrival if not provided
    # ------------------------------------------------------------
    if first_arrival_time is None:
        first_idx = np.argmax(np.abs(waveform))
        first_arrival_time = time[first_idx]
    else:
        first_idx = int(np.argmin(np.abs(time - first_arrival_time)))

    # ------------------------------------------------------------
    # Extract template around first arrival
    # ------------------------------------------------------------
    i0 = int(np.round((first_arrival_time - template_before) / dt))
    i1 = int(np.round((first_arrival_time + template_after) / dt))

    i0 = max(i0, 0)
    i1 = min(i1, n)

    template = waveform[i0:i1].copy()
    template_time = time[i0:i1] - first_arrival_time

    if len(template) < 3:
        raise ValueError("Template window is too short. Increase template_before/template_after.")

    template_norm = np.sqrt(np.sum(template**2))

    if template_norm == 0:
        raise ValueError("Template has zero energy. Choose a different first-arrival window.")

    n_template = len(template)
    corr = np.zeros_like(waveform)

    for start in range(0, n - n_template + 1):
        end = start + n_template

        segment = waveform[start:end].copy()

        corr[start] = np.sum(segment * template)

    corr = corr / np.max(np.abs(corr))
    corr_time = time.copy()

    # ------------------------------------------------------------
    # Find correlation peaks after the first arrival
    # ------------------------------------------------------------
    min_distance_samples = int(np.round(min_separation / dt))

    search_start_idx = int(np.round((first_arrival_time + template_after) / dt))
    search_start_idx = max(search_start_idx, 0)

    peaks, props = find_peaks(
        corr[search_start_idx:],
        height=min_corr,
        distance=min_distance_samples,
    )

    peaks = peaks + search_start_idx

    peak_times = corr_time[peaks]
    peak_corrs = corr[peaks]

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    if plot:
        mermaid_data = get_mermaid_data(data_file, data_index)
        name = mermaid_data["name"]
        stdp = mermaid_data["stdp"]
        ocdp = mermaid_data["ocdp"]
        fluid_arvl_angle = mermaid_data["water_angle_from_vertical_deg"]

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        # -------------------------
        # Top panel: waveform
        # -------------------------
        axes[0].plot(time, waveform, label="synthetic waveform")
        axes[0].axvspan(
            time[i0],
            time[i1 - 1],
            alpha=0.25,
            label="template window",
        )

        if t_arrivals is not None:
            ymin, ymax = axes[0].get_ylim()
            y_span = ymax - ymin
            y_label = ymin + 0.06 * y_span
            x_span = time[-1] - (-2)

            for i, ti in enumerate(t_arrivals):
                n_surface = int(np.ceil(i / 2))
                n_seafloor = int(np.floor(i / 2))
                label = rf"$w^{{s,b}}_{{{n_surface},{n_seafloor}}}$"

                if i == 0:
                    color = "k"
                elif i % 2 == 1:
                    color = "green"
                else:
                    color = "blue"

                axes[0].vlines(
                    ti,
                    ymin,
                    ymax,
                    colors=color,
                    linestyle="dashed",
                    linewidth=1.2,
                    alpha=0.8,
                )

                axes[0].annotate(
                    label,
                    xy=(ti, y_label),
                    xytext=(ti + 0.01 * x_span, y_label),
                    textcoords="data",
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    color=color,
                    rotation=0,
                )

        axes[0].grid(True)
        axes[0].set_ylabel("relative amplitude")
        axes[0].legend()
        axes[0].set_title(
            f"{name}\n"
            f"Arrival angle {fluid_arvl_angle:.2f}°, "
            f"Receiver Depth: {stdp} m, GEBCO Approximate Ocean Depth: {ocdp} m\n"
            f"First-arrival correlation with reverberations"
        )

        # -------------------------
        # Bottom panel: correlation
        # -------------------------
        axes[1].plot(corr_time, corr, label="normalized correlation")

        if t_arrivals is not None:
            ymin, ymax = axes[1].get_ylim()
            y_span = ymax - ymin
            y_label = ymin + 0.06 * y_span
            x_span = time[-1] - (-2)

            for i, ti in enumerate(t_arrivals):
                n_surface = int(np.ceil(i / 2))
                n_seafloor = int(np.floor(i / 2))
                label = rf"$w^{{s,b}}_{{{n_surface},{n_seafloor}}}$"

                if i == 0:
                    color = "k"
                elif i % 2 == 1:
                    color = "green"
                else:
                    color = "blue"

                axes[1].vlines(
                    ti,
                    ymin,
                    ymax,
                    colors=color,
                    linestyle="dashed",
                    linewidth=1.2,
                    alpha=0.8,
                )

                axes[1].annotate(
                    label,
                    xy=(ti, y_label),
                    xytext=(ti + 0.01 * x_span, y_label),
                    textcoords="data",
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    color=color,
                    rotation=0,
                )

        axes[1].grid(True)
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("normalized correlation")
        axes[1].set_ylim(-1.05, 1.05)
        axes[1].legend()

        axes[0].set_xlim(-2, time[-1])

        fig.tight_layout()
        plt.show()

    return corr, corr_time, peak_times, peak_corrs, template, template_time

if __name__ == "__main__":
    data_file = "./tomocat1.txt"
    data_index = 500
    t_max = 50.0
    dt = 0.01
    arrival_time_shift = 10.0
    noise_amp = 0.0
    source_func = gaussian_sinusoid
    window_length=5.0

    rng = np.random.default_rng(35)

    synthetic, h, time, w, tw, amp_arrivals, t_arrivals = source_time_spike_train(
        data_file=data_file,
        data_index=data_index,
        source_func=source_func,
        source_params={"f0": 1.0, "sigma": 1.0, "phase": 0.0},
        dt=dt,
        t_max=t_max,
        arrival_time_shift=arrival_time_shift,
        wavelet_width=8.0,
        noise_amp=noise_amp,
        noise_corner_freq=1.0,
        plot=False,
    )

    corr, corr_time, peak_times, peak_corrs, template, template_time = correlate_first_arrival(
        waveform=synthetic,
        time=time,
        first_arrival_time=arrival_time_shift,
        template_before=0.0,
        template_after=window_length,
        min_corr=0.3,
        min_separation=2.0,
        plot=True,
        amp_arrivals = amp_arrivals,
        t_arrivals=t_arrivals,
    )
