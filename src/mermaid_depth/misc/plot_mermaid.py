import numpy as np
import matplotlib.pyplot as plt
import obspy


def plot_mermaid_sac(
    sac_file,
    mermaid_data=None,
    bandpass=True,
    f1=0.5,
    f2=2.0,
    zerophase=True,
):
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

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        time,
        waveform,
        color="k",
        linewidth=1.0,
        label="MERMAID waveform",
    )

    ax.grid(True)
    ax.set_xlim(0, time[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("relative amplitude")
    ax.legend(loc="upper right")

    if mermaid_data is not None:
        name = mermaid_data["name"]
        H0 = mermaid_data["ocdp"]
        zf = mermaid_data["stdp"]
        theta_deg = mermaid_data["water_angle_from_vertical_deg"]

        title = (
            f"{name}\n"
            f"GEBCO H = {H0:.1f} m, "
            f"Receiver Depth = {zf:.1f} m, "
            f"Water Angle = {theta_deg:.2f}°"
        )
    else:
        title = f"MERMAID SAC: {sac_file}"

    if bandpass:
        title += f"\nBandpass: {f1:.2f}–{f2:.2f} Hz"

    ax.set_title(title)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    from mermaid_depth.misc.read_tomocat1 import get_mermaid_data

    file = "./tomocat1.txt"
    data_index = 318

    mermaid_data = get_mermaid_data(file, data_index)
    name = mermaid_data["name"]

    sac_file = f"./Mermaid_Data_Joel/{name}"

    plot_mermaid_sac(
        sac_file=sac_file,
        mermaid_data=mermaid_data,
        bandpass=True,
        f1=0.5,
        f2=2.0,
        zerophase=True,
    )