import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from mermaid_depth.depth_determination.fluidsolidarrivals import fluidsolidarrivals
from mermaid_depth.misc.read_tomocat1 import get_mermaid_data, read_tomocat1


"""GLOBAL CONSTANTS"""
RHO_F = 1020.0
VP_F = 1500.0
RHO_S = 2500.0
VP_S = 3400.0
VS_S = 1963.0

def spike_train(file, t_max, data_index, plot=False):
    mermaid_data = get_mermaid_data(file, data_index)

    name = mermaid_data["name"]
    ocdp = mermaid_data["ocdp"]
    stdp = mermaid_data["stdp"]
    fluid_arvl_angle = mermaid_data["water_angle_from_vertical_deg"]
    P_arvl_angle = mermaid_data["P_angle_from_vertical_deg"]

    t, x, z, ph = fluidsolidarrivals(
        z_interface=ocdp,
        z_station=stdp,
        rho_f = RHO_F,
        vp_f = VP_F,
        rho_s = RHO_S,
        vp_s = VP_S,
        vs_s = VS_S,
        theta = np.radians(P_arvl_angle),
        t_max = t_max
    )

    p_mask = np.abs(ph) == 1
    pressure_amp = np.real(x[p_mask])

    "Plot the spike train of acoustic pressure at the receiver."
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))

        t_plot = t[p_mask]
        pressure_plot = pressure_amp

        # Plot each spike individually so surface and seafloor reverberations
        # can be colored differently.
        for i, (ti, ai) in enumerate(zip(t_plot, pressure_plot)):
            if i == 0:
                color = "k"
            elif i % 2 == 1:
                color = "green"
            else:
                color = "blue"

            ax.vlines(ti, 0.0, ai, color=color, linewidth=1.5)
            ax.plot(ti, ai, "o", color=color, markersize=5)

        ax.axhline(0.0, linewidth=0.8)
        ax.set_xlim([-2, t_max])
        ax.set_ylim(1.1 * np.min(pressure_plot), 1.1 * np.max(pressure_plot))
        ax.grid(True)

        ax.set_xlabel("time (s)")
        ax.set_ylabel("relative pressure amplitude")
        ax.set_title(f"{name}\n Arrival angle {fluid_arvl_angle:.2f}°, Receiver Depth: {stdp} m, GEBCO Ocean Depth: {ocdp} m")

        x_span = t_max - (-2)

        for i, (ti, ai) in enumerate(zip(t_plot, pressure_plot)):
            n_surface = int(np.ceil(i / 2))
            n_seafloor = int(np.floor(i / 2))
            label = rf"$w^{{s,b}}_{{{n_surface},{n_seafloor}}}$"

            if i == 0:
                color = "k"
            elif i % 2 == 1:
                color = "green"
            else:
                color = "blue"

            ax.annotate(
                label,
                xy=(ti, ai),
                xytext=(ti + 0.015 * x_span, ai),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=9,
                color=color,
            )

        plt.show()

    return pressure_amp, t[p_mask]

if __name__ == "__main__":
    """INPUTS"""
    data_file= "./tomocat1.txt"
    t_max = 50
    data_index = 500

    spike_train(data_file, t_max, data_index, plot = True)
