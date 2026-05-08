import numpy as np
from typing import Any
from obspy.taup import TauPyModel
import math


def _to_float(value: str) -> float:
    return float(value)


def _to_int(value: str) -> int:
    return int(value)


def read_tomocat1(filename: str | None = None) -> dict[str, list[Any]]:
    """
    Read a MERMAID tomocat1 residual text file into a dictionary of columns.

    Parameters
    ----------
    filename : str
        Path to the updated residual formatted text file.

    Returns
    -------
    dict
        Dictionary whose keys mirror the MATLAB struct field names and whose
        values are lists containing one entry per row.
    """

    # Column names in the same order as the MATLAB code
    field_names = [
        "filename",
        "event_time",
        "evlo",
        "evla",
        "mag_val",
        "mag_type",
        "evdp",
        "seismogram_time",
        "stlo",
        "stla",
        "stdp",
        "ocdp",
        "gcarc_1D",
        "gcarc_1Dstar_adj",
        "gcarc_1Dstar",
        "gcarc_3D_adj",
        "gcarc_3D",
        "obs_travtime",
        "obs_arvltime",
        "travtime_1D",
        "arvltime_1D",
        "tres_1D",
        "travtime_1Dstar_adj",
        "travtime_1Dstar",
        "arvltime_1Dstar",
        "tres_1Dstar",
        "travtime_3D_adj",
        "travtime_3D",
        "arvltime_3D",
        "tres_3D",
        "twosd",
        "SNR",
        "max_counts",
        "max_delay",
        "NEIC_ID",
        "IRIS_ID",
        "KSTNM",
        "phase_name",
        "reviewer",
    ]

    # Types corresponding to the MATLAB textscan format
    converters = [
        str,        # filename
        str,        # event_time
        _to_float,  # evlo
        _to_float,  # evla
        _to_float,  # mag_val
        str,        # mag_type
        _to_float,  # evdp
        str,        # seismogram_time
        _to_float,  # stlo
        _to_float,  # stla
        _to_float,  # stdp
        _to_float,  # ocdp
        _to_float,  # gcarc_1D
        _to_float,  # gcarc_1Dstar_adj
        _to_float,  # gcarc_1Dstar
        _to_float,  # gcarc_3D_adj
        _to_float,  # gcarc_3D
        _to_float,  # obs_travtime
        _to_float,  # obs_arvltime
        _to_float,  # travtime_1D
        _to_float,  # arvltime_1D
        _to_float,  # tres_1D
        _to_float,  # travtime_1Dstar_adj
        _to_float,  # travtime_1Dstar
        _to_float,  # arvltime_1Dstar
        _to_float,  # tres_1Dstar
        _to_float,  # travtime_3D_adj
        _to_float,  # travtime_3D
        _to_float,  # arvltime_3D
        _to_float,  # tres_3D
        _to_float,  # twosd
        _to_float,  # SNR
        _to_float,  # max_counts
        _to_float,  # max_delay
        str,        # NEIC_ID
        str,        # IRIS_ID
        str,        # KSTNM
        str,        # phase_name
        _to_int,    # reviewer
    ]

    data: dict[str, list[Any]] = {name: [] for name in field_names}

    with open(filename, "r", encoding="utf-8") as f:
        # Skip the first two header lines
        next(f, None)
        next(f, None)

        for line_num, line in enumerate(f, start=3):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()  # handles multiple spaces like MATLAB option
            if len(parts) != len(field_names):
                raise ValueError(
                    f"Line {line_num}: expected {len(field_names)} columns, "
                    f"found {len(parts)}.\nLine content: {line!r}"
                )

            for name, conv, value in zip(field_names, converters, parts):
                try:
                    data[name].append(conv(value))
                except Exception as exc:
                    raise ValueError(
                        f"Line {line_num}: failed to parse field '{name}' "
                        f"with value {value!r}"
                    ) from exc

    # Add 0:360 longitude versions
    data["evlo_360"] = [x if x >= 0 else x + 360 for x in data["evlo"]]
    data["stlo_360"] = [x if x >= 0 else x + 360 for x in data["stlo"]]

    return data

def taup_to_water_angle(
    evdp,
    gcarc_deg,
    phase_name,
    ocdp,
    c_water=1500,
    c_P=3400,
    model_name="iasp91",
):
    R_EARTH_KM = 6371.0
    model = TauPyModel(model=model_name)
    arrivals = model.get_travel_times(
        source_depth_in_km=evdp/1000,
        distance_in_degree=gcarc_deg,
        phase_list=[phase_name, phase_name.upper(), phase_name.lower()],
    )
    if not arrivals:
        return None

    arr = arrivals[0]

    # TauP ray parameter: s/radian
    p_taup = arr.ray_param

    # Convert to local horizontal slowness at seafloor: s/km
    r_sf = R_EARTH_KM - ocdp/1000
    p_h = p_taup / r_sf

    arg = p_h * c_water/1000
    if abs(arg) > 1:
        raise ValueError(
            f"p*c_water = {arg:.4f} > 1, check units/model/phase selection."
        )

    theta_w_rad = np.arcsin(arg)
    theta_w_deg = np.degrees(theta_w_rad)

    arg2 = p_h * c_P/1000
    if abs(arg2) > 1:
        raise ValueError(
            f"p*c_P = {arg2:.4f} > 1, check units/model/phase selection."
        )
    
    theta_s_rad = np.arcsin(arg2)
    theta_s_deg = np.degrees(theta_s_rad)

    return {
        "phase": arr.name,
        "travel_time_s": arr.time,
        "taup_ray_param_s_per_rad": p_taup,
        "horizontal_slowness_s_per_km": p_h,
        "taup_incident_angle_deg": arr.incident_angle,
        "water_angle_from_vertical_deg": theta_w_deg,
        "P_angle_from_vertical_deg": theta_s_deg,
    }

def get_mermaid_data(data_file, data_index, VP_F=1500, VP_S=3400):
    """
    Return the organized MERMAID metadata dictionary for one valid TOMOCAT row.

    data_index is counted only among rows with valid observed arrival times,
    matching the convention used in spike_train/source_time_spike_train.
    """

    mermaid_all = read_tomocat1(data_file)

    valid_indices = [
        i for i, t in enumerate(mermaid_all["obs_arvltime"])
        if t is not None and not math.isnan(t)
    ]

    valid_index = valid_indices[data_index]

    ocdp = mermaid_all["ocdp"][valid_index]
    evdp = mermaid_all["evdp"][valid_index]
    gcarc_1D = mermaid_all["gcarc_1D"][valid_index]
    phase_name = mermaid_all["phase_name"][valid_index]

    angle_info = taup_to_water_angle(
        evdp=evdp,
        gcarc_deg=gcarc_1D,
        phase_name=phase_name,
        ocdp=ocdp,
        c_water=VP_F,
        c_P=VP_S,
        model_name="iasp91",
    )

    mermaid_data = {
        "valid_index": valid_index,
        "name": mermaid_all["filename"][valid_index],
        "ocdp": ocdp,
        "evdp": evdp,
        "stdp": mermaid_all["stdp"][valid_index],
        "stlo": mermaid_all["stlo"][valid_index],
        "stla": mermaid_all["stla"][valid_index],
        "gcarc_1D": gcarc_1D,
        "phase_name": phase_name,
        "obs_arvltime": mermaid_all["obs_arvltime"][valid_index],
        "obs_travtime": mermaid_all["obs_travtime"][valid_index],
        "water_angle_from_vertical_deg": angle_info["water_angle_from_vertical_deg"],
        "P_angle_from_vertical_deg": angle_info["P_angle_from_vertical_deg"],
        "taup_travel_time_s": angle_info["travel_time_s"],
        "taup_ray_param_s_per_rad": angle_info["taup_ray_param_s_per_rad"],
        "horizontal_slowness_s_per_km": angle_info["horizontal_slowness_s_per_km"],
    }

    return mermaid_data

if __name__ == "__main__":
    data_file = "./tomocat1.txt"
    data = read_tomocat1(data_file)

    data_index = 141
    print(f"Data for index {data_index}:")
    mermaid_data = get_mermaid_data(data_file, data_index)
    for key, value in mermaid_data.items():
        print(f"{key}: {value}")

    print('Length of each column:')
    for key, column in data.items():
        print(f"{key}: {len(column)}")