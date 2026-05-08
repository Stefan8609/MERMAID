import numpy as np
import matplotlib.pyplot as plt
import obspy
from mermaid_depth.depth_determination.waveform_stacking import (
    grid_search_ocean_depth,
    plot_depth_search,
)
from mermaid_depth.depth_determination.waveform_stacking_2d import (
    grid_search_ocean_depth2D,
    plot_depth_search2D,
)
from mermaid_depth.misc.read_tomocat1 import (
    get_mermaid_data,
)


def load_mermaid_sac(
    sac_file,
    bandpass=True,
    f1=0.5,
    f2=2.0,
    zerophase=True,
    window_start = None,
    window_end = None,
):
    """
    Load and lightly preprocess a real MERMAID SAC waveform.

    Returns
    -------
    waveform : ndarray
        Processed waveform.
    time : ndarray
        Time vector in seconds relative to SAC start.
    tr : obspy.Trace
        ObsPy trace, including SAC headers.
    """

    st = obspy.read(sac_file)
    tr = st[0]

    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=0.05)

    if bandpass:
        tr.filter(
            "bandpass",
            freqmin=f1,
            freqmax=f2,
            corners=4,
            zerophase=zerophase,
        )

    waveform = tr.data.astype(float)
    dt = tr.stats.delta
    time = np.arange(len(waveform)) * dt

    if window_start is not None and window_end is not None:
        # Apply windowing to isolate the desired time range
        window_mask = (time >= window_start) & (time <= window_end)
        waveform = waveform[window_mask]
        time = time[window_mask]

    # Normalize only globally for plotting/search stability.
    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / max_amp

    return waveform, time, tr

def find_depth(
    data_index,
    H_radius = 250,
    H_step = 1,
    T_radius = 1.0,
    T_step = 0.1,
    packet_before=0.0,
    packet_pad_after=2.0,
    demean_windows=False,
    dim=1,
    plot=False,
):
    """
    Find the ocean depth H that best matches the observed first arrival time.

    Returns
    -------
    best_H : float
        The ocean depth from H_grid that minimizes the misfit.
    """

    file = "./tomocat1.txt"
    mermaid_data = get_mermaid_data(file, data_index)

    window_start = mermaid_data["obs_arvltime"] - 10.0
    window_end = mermaid_data["obs_arvltime"] + 40.0

    name = mermaid_data["name"]
    H0 = mermaid_data["ocdp"]
    zf = mermaid_data["stdp"]
    obs_arvltime = mermaid_data["obs_arvltime"]
    theta_deg = mermaid_data["water_angle_from_vertical_deg"]
    theta = np.radians(theta_deg)

    waveform, time, _ = load_mermaid_sac(f"./Mermaid_Data_Joel/{name}",         
                                        window_start=window_start, 
                                        window_end=window_end
                        )

    H_min = max(zf + 10.0, H0 - H_radius)
    H_max = H0 + H_radius
    H_grid = np.arange(H_min, H_max + H_step, H_step)

    if dim == 1:
        search_result = grid_search_ocean_depth(
            waveform=waveform,
            time=time,
            first_arrival_time=obs_arvltime,
            H_grid=H_grid,
            zf=zf,
            theta=theta,
            packet_before=packet_before,
            packet_pad_after=packet_pad_after,
            demean_windows=demean_windows,
        )
        if plot:
            plot_depth_search(
                waveform=waveform,
                time=time,
                search_result=search_result,
                data_file= file,
                data_index=data_index,
                # first_arrival_time=obs_arvltime,
            )
    
    if dim == 2:
        first_arrival_time_grid = np.arange(
            obs_arvltime - T_radius,
            obs_arvltime + T_radius + T_step,
            T_step,
        )

        search_result = grid_search_ocean_depth2D(
            waveform=waveform,
            time=time,
            first_arrival_time_grid=first_arrival_time_grid,
            H_grid=H_grid,
            zf=zf,
            theta=theta,
            packet_before=packet_before,
            packet_pad_after=packet_pad_after,
            demean_windows=demean_windows,
        )

        if plot:
            plot_depth_search2D(
                waveform=waveform,
                time=time,
                search_result=search_result,
                data_file=file,
                data_index=data_index,
                first_arrival_time_ref=obs_arvltime,
                t_arrivals=None,
            )

    return search_result

if __name__ == "__main__":
    # import warnings
    # import numpy as np

    # warnings.filterwarnings(
    #     "ignore",
    #     message=r"Sample spacing read from SAC file.*was rounded.*microsecond precision",
    #     category=UserWarning,
    #     module=r"obspy\.io\.sac\.util",
    # )

    # file = "./tomocat1.txt"
    # save_path = "./saved_data/depth_search_results.npz"

    # data_indices = []
    # found_H_values = []
    # ocdp_values = []
    # depth_differences = []
    # best_scores = []
    # best_coherences = []
    # n_valid_values = []

    # failed_data_indices = []
    # failed_errors = []

    # for data_index in range(0, 12091, 1):
    #     try:
    #         result = find_depth(
    #             data_index=data_index,
    #             H_radius=500,
    #             H_step=1,
    #             T_radius=1.0,
    #             T_step=0.1,
    #             packet_before=0.0,
    #             packet_pad_after=2.0,
    #             demean_windows=False,
    #             dim=1,
    #             plot=False,
    #         )

    #         mermaid_data = get_mermaid_data(file, data_index)

    #         best = result["best"]
    #         found_H = best["H"]
    #         ocdp = mermaid_data["ocdp"]
    #         depth_difference = found_H - ocdp

    #         print(
    #             f"Data index: {data_index}, "
    #             f"Found H: {found_H:.1f} m, "
    #             f"OC DP: {ocdp:.1f} m, "
    #             f"Difference: {depth_difference:.1f} m"
    #         )

    #         data_indices.append(data_index)
    #         found_H_values.append(found_H)
    #         ocdp_values.append(ocdp)
    #         depth_differences.append(depth_difference)

    #         best_scores.append(best.get("score", np.nan))
    #         best_coherences.append(best.get("coherence", np.nan))
    #         n_valid_values.append(best.get("n_valid", np.nan))

    #     except Exception as err:
    #         print(f"Data index: {data_index}, skipped: {err}")
    #         failed_data_indices.append(data_index)
    #         failed_errors.append(str(err))
    #         continue

    # np.savez_compressed(
    #     save_path,
    #     data_indices=np.asarray(data_indices, dtype=int),
    #     found_H=np.asarray(found_H_values, dtype=float),
    #     ocdp=np.asarray(ocdp_values, dtype=float),
    #     depth_differences=np.asarray(depth_differences, dtype=float),
    #     best_scores=np.asarray(best_scores, dtype=float),
    #     best_coherences=np.asarray(best_coherences, dtype=float),
    #     n_valid=np.asarray(n_valid_values, dtype=float),
    #     failed_data_indices=np.asarray(failed_data_indices, dtype=int),
    #     failed_errors=np.asarray(failed_errors, dtype=object),
    # )

    # print(f"\nSaved results to: {save_path}")
    # print(f"Successful searches: {len(data_indices)}")
    # print(f"Failed searches: {len(failed_data_indices)}")

    data_index = 1062
    result = find_depth(
        data_index=data_index,
        H_radius=500,
        H_step=1,
        T_radius=1.0,
        T_step=0.1,
        packet_before=0.0,
        packet_pad_after=2.0,
        demean_windows=False,
        dim=1,
        plot=True,
    )