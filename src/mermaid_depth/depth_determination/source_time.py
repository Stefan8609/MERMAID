import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from mermaid_depth.depth_determination.spike_train import spike_train
from mermaid_depth.misc.read_tomocat1 import get_mermaid_data


def gaussian_sinusoid(t, f0=1.0, sigma=1.0, phase=0.0):
    peak_time = 2.0 * sigma
    envelope = np.exp(-0.5 * ((t - peak_time) / sigma) ** 2)
    return envelope * np.sin(2.0 * np.pi * f0 * t + phase)

def triangle_wavelet(t, width=1.0):
    y = 1.0 - np.abs(t) / (0.5 * width)
    y[y < 0.0] = 0.0
    return y


def square_wavelet(t, width=1.0):
    return np.where(np.abs(t) <= 0.5 * width, 1.0, 0.0)


def ricker_wavelet(t, f0=2.0):
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def smooth_white_noise(time, dt, noise_std, corner_freq=1.0, seed=42, pad_seconds=20.0):
    rng = np.random.default_rng(seed)

    n = len(time)
    pad_n = int(round(pad_seconds / dt))
    n_long = n + 2 * pad_n

    white = rng.normal(0.0, 1.0, n_long)

    fs = 1.0 / dt
    nyq = 0.5 * fs

    b, a = butter(
        4,
        corner_freq / nyq,
        btype="lowpass",
    )

    noise_long = filtfilt(b, a, white)

    # Cut out the middle to remove beginning/end filter artifacts
    noise = noise_long[pad_n:pad_n + n]

    noise -= np.mean(noise)
    std = np.std(noise)
    if std > 0:
        noise /= std

    noise *= noise_std

    return noise

def source_time_spike_train(
    data_file,
    data_index,
    source_func = None,
    source_params=None,
    dt=0.01,
    t_max=100.0,
    arrival_time_shift = 10.0,
    wavelet_width=2.0,
    noise_amp=0.1,
    noise_corner_freq=1.0,
    plot=False,
):
    if source_params is None:
        source_params = {}

    # Sparse arrivals from your existing function
    amp_arrivals, t_arrivals = spike_train(
        data_file,
        t_max=t_max,
        data_index=data_index,
        plot=False,
    )

    amp_arrivals = np.asarray(amp_arrivals, dtype=float)
    t_arrivals = np.asarray(t_arrivals, dtype=float)

    # Move entire spike train later in the waveform
    t_arrivals = t_arrivals + arrival_time_shift

    # Uniform time axis
    time = np.arange(0.0, t_max + dt, dt)

    # Sample sparse arrivals onto uniform grid
    h = np.zeros_like(time)
    for ti, ai in zip(t_arrivals, amp_arrivals):
        idx = int(np.round(ti / dt))
        if 0 <= idx < len(h):
            h[idx] += ai

    # Build source-time function on a zero-centered time axis
    if source_func:
        tw = np.arange(0.0, wavelet_width + dt, dt)
        wavelet = source_func(tw, **source_params)

        # Normalize source wavelet
        max_abs = np.max(np.abs(wavelet))
        if max_abs > 0:
            wavelet = wavelet / max_abs

        # Always use same-length convolution
        synthetic = np.convolve(h, wavelet, mode="full")[: len(time)]
    else:
        wavelet = None
        tw = None
        synthetic = h

    # Add low-pass filtered white noise
    noise = smooth_white_noise(
    time=time,
    dt=dt,
    noise_std=noise_amp * np.max(np.abs(synthetic)),
    corner_freq=noise_corner_freq,
    seed=42,
    )

    synthetic_noisy = synthetic + noise
    
    # Normalize final synthetic waveform for better visualization
    max_abs_synth = np.max(np.abs(synthetic_noisy))
    synthetic_noisy = synthetic_noisy / max_abs_synth

    if plot:
        mermaid_data = get_mermaid_data(data_file, data_index)
        name = mermaid_data["name"]
        stdp = mermaid_data["stdp"]
        ocdp = mermaid_data["ocdp"]
        fluid_arvl_angle = mermaid_data["water_angle_from_vertical_deg"]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(time, synthetic_noisy, label="synthetic waveform")

        ax.set_xlim(-2, t_max)
        ax.grid(True)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("relative amplitude")
        ax.legend()
        ax.set_title(f"{name}\n Arrival angle {fluid_arvl_angle:.2f}°, Receiver Depth: {stdp} m, GEBCO Approximate Ocean Depth: {ocdp} m \n Gaussian Sinusoid Source Time Function with f={source_params.get('f0', 2.0)} Hz, sigma={source_params.get('sigma', 0.5)} s")

        ymin, ymax = ax.get_ylim()
        y_span = ymax - ymin
        y_label = ymin + 0.06 * y_span
        x_span = t_max - (-2)

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

            ax.vlines(
                ti,
                ymin,
                ymax,
                colors=color,
                linestyle="dashed",
                linewidth=1.2,
                alpha=0.8,
            )

            ax.annotate(
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

        fig.tight_layout()
        plt.show()

    return synthetic_noisy, h, time, wavelet, tw, amp_arrivals, t_arrivals

if __name__ == "__main__":
    data_file = "./tomocat1.txt"
    data_index = 500
    t_max = 50.0
    dt = 0.01
    noise_amp = 0.0
    noise_corner_freq = 1.0
    arrival_time_shift = 10.0
    source_func = gaussian_sinusoid

    #Repeatable random noise generator for testing purposes
    rng = np.random.default_rng(35)

    synthetic, h, time, w, tw, _, _ = source_time_spike_train(
        data_file=data_file,
        data_index=data_index,
        source_func=source_func,
        source_params={"f0":1.0, "sigma":1.0, "phase": 0},
        dt=dt,
        t_max=t_max,
        arrival_time_shift=arrival_time_shift,
        wavelet_width=8.0,
        noise_amp=noise_amp,
        noise_corner_freq=noise_corner_freq,
        plot=True,
    )
