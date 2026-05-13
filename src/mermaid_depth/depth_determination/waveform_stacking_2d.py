import numpy as np
import matplotlib.pyplot as plt
from mermaid_depth.depth_determination.source_time import (
    gaussian_sinusoid,
    source_time_spike_train,
)
from mermaid_depth.misc.read_tomocat1 import get_mermaid_data

"Huang+2015 paper inspired"

"""GLOBAL CONSTANTS"""
VP_F = 1500.0  # m/s, water/acoustic velocity
N_SEAFLOOR_REVERBS = 4
MIN_VALID_REVERBS = 2


def correlate_spikes(
    waveform,
    time,
    first_arrival_time,
    H,
    zf,
    theta,
    packet_before,
    packet_after,
    demean_windows=True,
):
    """
    For one trial ocean depth H and first-arrival time:

    1. Predict seafloor reverberation packet starts.
    2. Put the surface echo inside each packet window.
    3. Polarity-correct each packet.
    4. Stack the packets.
    5. Return the score and only the best-packet data needed for plotting/diagnostics.

    Larger score is better.
    """

    dt = float(np.median(np.diff(time)))
    q_w = np.cos(theta) / VP_F
    surface_delay = 2.0 * zf * q_w

    n_before = int(np.round(packet_before / dt))
    n_after = int(np.round(packet_after / dt))
    rel_time = np.arange(-n_before, n_after + 1) * dt

    windows = []
    bottom_times = []
    surface_times = []
    polarities = []

    for n in range(0, N_SEAFLOOR_REVERBS + 1):
        bottom_time = first_arrival_time + 2.0 * n * H * q_w
        surface_time = bottom_time + surface_delay
        sample_time = bottom_time + rel_time

        if sample_time[0] < time[0] or sample_time[-1] > time[-1]:
            continue

        window = np.interp(sample_time, time, waveform)

        if demean_windows:
            window = window - np.mean(window)

        # One full water-column loop has one surface and one bottom reflection.
        polarity = (-1.0) ** n

        windows.append(polarity * window)
        bottom_times.append(bottom_time)
        surface_times.append(surface_time)
        polarities.append(polarity)

    if len(windows) < MIN_VALID_REVERBS:
        return None

    windows = np.asarray(windows)
    stack = np.mean(windows, axis=0)

    stack_energy = np.sum(stack**2)

    vals = []
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            a = windows[i]
            b = windows[j]
            denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
            if denom > 0.0:
                vals.append(np.sum(a * b) / denom)

    coherence = float(np.mean(vals)) if vals else 0.0
    score = stack_energy * max(coherence, 0.0)

    return {
        "H": H,
        "first_arrival_time": first_arrival_time,
        "score": score,
        "coherence": coherence,
        "n_valid": len(windows),
        "rel_time": rel_time,
        "windows": windows,
        "stack": stack,
        "bottom_times": np.asarray(bottom_times),
        "surface_times": np.asarray(surface_times),
        "polarities": np.asarray(polarities),
        "surface_delay": surface_delay,
    }


def grid_search_ocean_depth2D(
    waveform,
    time,
    first_arrival_time_grid,
    H_grid,
    zf,
    theta,
    packet_before,
    packet_pad_after,
    demean_windows=True,
):
    """
    2D grid search over ocean depth H and first-arrival time t0.

    Returns the same dictionary structure as grid_search_ocean_depth. scores has
    shape:

        (len(first_arrival_time_grid), len(H_grid))

    Larger score is better.
    """

    H_grid = np.asarray(H_grid, dtype=float)
    first_arrival_time_grid = np.asarray(first_arrival_time_grid, dtype=float)

    scores = np.full((len(first_arrival_time_grid), len(H_grid)), np.nan)
    best = None
    best_score = -np.inf

    q_w = np.cos(theta) / VP_F
    surface_delay = 2.0 * zf * q_w
    packet_after = surface_delay + packet_pad_after

    for it0, first_arrival_time in enumerate(first_arrival_time_grid):
        for iH, H in enumerate(H_grid):
            if H <= zf:
                continue

            result = correlate_spikes(
                waveform=waveform,
                time=time,
                first_arrival_time=first_arrival_time,
                H=H,
                zf=zf,
                theta=theta,
                packet_before=packet_before,
                packet_after=packet_after,
                demean_windows=demean_windows,
            )

            if result is None:
                continue

            scores[it0, iH] = result["score"]

            if result["score"] > best_score:
                best_score = result["score"]
                best = result

    if best is None:
        raise ValueError(
            "No valid depth found. Try fewer reverbs, larger t_max, "
            "a smaller H range, or shorter windows."
        )

    return {
        "H_grid": H_grid,
        "first_arrival_time_grid": first_arrival_time_grid,
        "scores": scores,
        "best": best,
    }


def plot_depth_search2D(
    waveform,
    time,
    search_result,
    data_file,
    data_index,
    first_arrival_time_ref,
):
    """
    Plot waveform, best stacked packets, and Huang-style 2D grid-search score.

    The contour plot is now:

        x-axis: trial ocean depth H
        y-axis: trial first-arrival time t0
    """

    best = search_result["best"]
    H_grid = search_result["H_grid"]
    first_arrival_time_grid = search_result["first_arrival_time_grid"]
    scores = search_result["scores"]

    mermaid_data = get_mermaid_data(data_file, data_index)

    H0 = mermaid_data["ocdp"]
    zf = mermaid_data["stdp"]
    name = mermaid_data["name"]
    theta_deg = mermaid_data["water_angle_from_vertical_deg"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    # ------------------------------------------------------------
    # Panel 1: waveform
    # ------------------------------------------------------------
    axes[0].plot(time, waveform, label="synthetic waveform")

    axes[0].axvline(
        first_arrival_time_ref,
        linestyle="--",
        linewidth=1.0,
        color="gray",
        label="reference first arrival",
    )

    axes[0].axvline(
        best["first_arrival_time"],
        linestyle="--",
        linewidth=1.0,
        color="k",
        label="best first arrival",
    )

    axes[0].grid(True)
    axes[0].set_ylabel("relative amplitude")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"{name}\n"
        f"Best H = {best['H']:.1f} m, "
        f"GEBCO H = {H0:.1f} m, "
        f"Best t0 = {best['first_arrival_time']:.3f} s, "
        f"Reference t0 = {first_arrival_time_ref:.3f} s, \n"
        f"Receiver Depth = {zf:.1f} m, "
        f"Water Angle = {theta_deg:.2f}°"
    )

    ymin, ymax = axes[0].get_ylim()
    y_span = ymax - ymin
    x_span = time[-1] - time[0]

    y_label_direct = ymin + 0.05 * y_span
    y_label_bottom = ymin + 0.13 * y_span
    y_label_surface = ymin + 0.21 * y_span

    # Direct arrival for the best-fitting coherence search.
    # This corresponds to best["bottom_times"][0], but should be labeled only
    # as the direct arrival, not as a seafloor reverberation.
    axes[0].text(
        best["first_arrival_time"] + 0.01 * x_span,
        y_label_direct,
        r"$w^{s,b}_{0,0}$",
        rotation=0,
        ha="left",
        va="bottom",
        fontsize=8,
        color="k",
    )

    # Seafloor/bottom reverberation packet starts.
    # Skip n = 0 because best["bottom_times"][0] is the direct arrival.
    for n, bottom_time in enumerate(best["bottom_times"]):
        if n == 0:
            continue

        axes[0].axvline(
            bottom_time,
            linestyle="--",
            linewidth=0.9,
            color="blue",
            alpha=0.8,
        )
        axes[0].text(
            bottom_time + 0.01 * x_span,
            y_label_bottom,
            rf"$w^{{s,b}}_{{{n},{n}}}$",
            rotation=0,
            ha="left",
            va="bottom",
            fontsize=8,
            color="blue",
        )

    # Surface reverberations.
    # For n = 0, this is the first surface reverberation: w^{s,b}_{1,0}.
    for n, surface_time in enumerate(best["surface_times"]):
        axes[0].axvline(
            surface_time,
            linestyle="--",
            linewidth=0.9,
            color="green",
            alpha=0.8,
        )
        axes[0].text(
            surface_time + 0.01 * x_span,
            y_label_surface,
            rf"$w^{{s,b}}_{{{n + 1},{n}}}$",
            rotation=0,
            ha="left",
            va="bottom",
            fontsize=8,
            color="green",
        )

    # ------------------------------------------------------------
    # Panel 2: best packet stack
    # ------------------------------------------------------------
    for window in best["windows"]:
        axes[1].plot(best["rel_time"], window, alpha=0.4)

    axes[1].plot(
        best["rel_time"],
        best["stack"],
        linewidth=2.0,
        label="stacked ocean reverberations",
    )

    axes[1].axvline(
        0.0,
        linestyle="--",
        linewidth=0.8,
        color="blue",
        label="seafloor packet start",
    )
    axes[1].axvline(
        best["surface_delay"],
        linestyle="--",
        linewidth=0.8,
        color="green",
        label="surface echo inside packet",
    )

    axes[1].grid(True)
    axes[1].set_xlabel("time relative to seafloor packet start (s)")
    axes[1].set_ylabel("polarity-corrected amplitude")
    axes[1].legend()

    # ------------------------------------------------------------
    # Panel 3: Huang-style 2D contour
    # ------------------------------------------------------------
    score_max = np.nanmax(scores)
    if not np.isfinite(score_max) or score_max <= 0.0:
        raise ValueError("Cannot normalize scores because no positive finite score was found.")

    score_norm = scores / score_max

    H_mesh, t0_mesh = np.meshgrid(H_grid, first_arrival_time_grid)

    contour = axes[2].contourf(
        H_mesh,
        t0_mesh,
        score_norm,
        levels=40,
    )

    fig.colorbar(
        contour,
        ax=axes[2],
        label="normalized stack score",
    )

    axes[2].plot(
        best["H"],
        best["first_arrival_time"],
        "w+",
        markersize=14,
        markeredgewidth=2.5,
        label="best fit",
    )

    axes[2].axvline(
        H0,
        color="w",
        linestyle=":",
        linewidth=1.5,
        label="GEBCO H",
    )

    axes[2].axhline(
        first_arrival_time_ref,
        color="w",
        linestyle="--",
        linewidth=1.5,
        label="reference t0",
    )

    axes[2].set_xlabel("trial ocean depth H (m)")
    axes[2].set_ylabel("trial first-arrival time t0 (s)")
    axes[2].set_title("2D grid search: ocean depth and first-arrival time")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    """INPUTS"""
    data_file = "./tomocat1.txt"
    data_index = 500

    t_max = 50.0
    dt = 0.01
    arrival_time_shift = 10.0

    rng = np.random.default_rng(35)

    # ------------------------------------------------------------
    # Gaussian synthetic tester
    # ------------------------------------------------------------
    source_func = gaussian_sinusoid
    source_params = {
        "f0": 1.0,
        "sigma": 1.0,
        "phase": 0.0,
    }

    wavelet_width = 8.0
    noise_amp = 0.2

    synthetic, h, time, w, tw, amp_arrivals, t_arrivals = source_time_spike_train(
        data_file=data_file,
        data_index=data_index,
        source_func=source_func,
        source_params=source_params,
        dt=dt,
        t_max=t_max,
        arrival_time_shift=arrival_time_shift,
        wavelet_width=wavelet_width,
        noise_amp=noise_amp,
        plot=False,
    )

    mermaid_data = get_mermaid_data(data_file, data_index)

    H0 = mermaid_data["ocdp"]
    zf = mermaid_data["stdp"]
    theta_deg = mermaid_data["water_angle_from_vertical_deg"]
    theta = np.radians(theta_deg)

    print("\nMERMAID metadata")
    print("----------------")
    print(f"Name: {mermaid_data['name']}")
    print(f"GEBCO ocean depth H0: {H0:.1f} m")
    print(f"Receiver depth zf: {zf:.1f} m")
    print(f"Water angle from vertical: {theta_deg:.2f} deg")

    # ------------------------------------------------------------
    # H-grid
    # ------------------------------------------------------------
    H_radius = 500.0
    H_step = 1.0

    H_min = max(zf + 10.0, H0 - H_radius)
    H_max = H0 + H_radius
    H_grid = np.arange(H_min, H_max + H_step, H_step)

    # ------------------------------------------------------------
    # First-arrival-time grid
    # ------------------------------------------------------------
    t0_radius = 0.5
    t0_step = dt

    first_arrival_time_grid = np.arange(
        arrival_time_shift - t0_radius,
        arrival_time_shift + t0_radius + t0_step,
        t0_step,
    )

    # ------------------------------------------------------------
    # Gaussian/window settings
    # ------------------------------------------------------------
    packet_before = 2.0
    packet_pad_after = 4.0
    demean_windows = True

    # ------------------------------------------------------------
    # Run 2D grid search over H and first-arrival time
    # ------------------------------------------------------------
    search_result = grid_search_ocean_depth2D(
        waveform=synthetic,
        time=time,
        first_arrival_time_grid=first_arrival_time_grid,
        H_grid=H_grid,
        zf=zf,
        theta=theta,
        packet_before=packet_before,
        packet_pad_after=packet_pad_after,
        demean_windows=demean_windows,
    )

    best = search_result["best"]

    print("\nDepth / first-arrival-time search result")
    print("----------------------------------------")
    print(f"Best-fit H: {best['H']:.1f} m")
    print(f"GEBCO H: {H0:.1f} m")
    print(f"Difference best H - GEBCO H: {best['H'] - H0:.1f} m")

    print(f"Best-fit first arrival time: {best['first_arrival_time']:.3f} s")
    print(f"Reference first arrival time: {arrival_time_shift:.3f} s")
    print(
        "Difference best t0 - reference t0: "
        f"{best['first_arrival_time'] - arrival_time_shift:.3f} s"
    )

    print(f"Best score: {best['score']:.6g}")
    print(f"Best packet coherence: {best['coherence']:.3f}")
    print(f"Number of valid seafloor packets: {best['n_valid']}")
    print(f"Internal surface delay: {best['surface_delay']:.3f} s")

    print("\nBest predicted packet times")
    print("---------------------------")
    for i, (tb, ts, pol) in enumerate(
        zip(best["bottom_times"], best["surface_times"], best["polarities"]),
        start=1,
    ):
        print(
            f"B{i}: bottom packet at {tb:.3f} s, "
            f"surface echo inside packet at {ts:.3f} s, "
            f"polarity correction {pol:+.0f}"
        )

    plot_depth_search2D(
        waveform=synthetic,
        time=time,
        search_result=search_result,
        data_file=data_file,
        data_index=data_index,
        first_arrival_time_ref=arrival_time_shift,
    )
