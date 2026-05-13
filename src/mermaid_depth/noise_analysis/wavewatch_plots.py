import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mermaid_depth.noise_analysis.wavewatch_gfs import gfswave_field

def _get_varname(ds, varname=None):
    """Return the requested xarray data variable name, or the first one."""
    if varname is not None:
        return varname

    data_vars = list(ds.data_vars)
    if len(data_vars) == 0:
        raise ValueError("Dataset has no data variables.")

    return data_vars[0]


def _get_valid_time(ds):
    """
    Try to extract a valid time from an xarray Dataset produced by Herbie/cfgrib.
    """
    for name in ["valid_time", "time"]:
        if name in ds.coords:
            value = ds.coords[name].values

            if np.ndim(value) == 0:
                return pd.to_datetime(value.item())

            return pd.to_datetime(value[0])

    if "requested_valid_time" in ds.attrs:
        return pd.to_datetime(ds.attrs["requested_valid_time"])

    return pd.NaT


def plot_wave_height_map(ds, lat0=None, lon0=None, varname=None):
    """
    Plot a full GFS-Wave xarray field.

    Parameters
    ----------
    ds : xarray.Dataset
        Full gridded GFS-Wave field.
    lat0, lon0 : float, optional
        Optional marker location, e.g. MERMAID position.
    varname : str, optional
        Data variable name. If None, uses the first data variable.
    """

    varname = _get_varname(ds, varname)
    da = ds[varname].squeeze()

    plt.figure(figsize=(9, 6))

    da.plot(
        x="longitude",
        y="latitude",
        cmap="viridis",
        cbar_kwargs={"label": varname},
    )

    if lat0 is not None and lon0 is not None:
        plt.scatter(
            lon0 % 360,
            lat0,
            marker="*",
            s=180,
            color="red",
            edgecolor="black",
            linewidth=0.8,
            label="MERMAID",
        )
        plt.legend()

    valid_time = _get_valid_time(ds)
    if pd.notna(valid_time):
        plt.title(f"GFS-Wave {varname}\nValid time: {valid_time}")
    else:
        plt.title(f"GFS-Wave {varname}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()


def extract_nearest_point_record(ds, lat, lon, varname=None, output_name=None):
    """
    Extract the nearest grid point from a full xarray field and return
    a one-row dictionary suitable for building a DataFrame.
    """

    varname = _get_varname(ds, varname)
    output_name = output_name or varname

    point = ds.sel(
        latitude=lat,
        longitude=lon % 360,
        method="nearest",
    )

    value = float(np.asarray(point[varname].values).squeeze())

    return {
        "valid_time": _get_valid_time(ds),
        "lat_requested": lat,
        "lon_requested": lon,
        "lat_model": float(point.latitude.values),
        "lon_model": float(point.longitude.values),
        output_name: value,
    }


def plot_wavewatch_timeseries(df, event_time=None):
    """
    Plot point time series extracted from xarray fields.

    df should contain columns like:
    valid_time, HTSGW, PERPW, DIRPW
    """

    df = df.copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    if "HTSGW" in df:
        axes[0].plot(df["valid_time"], df["HTSGW"], marker="o")
        axes[0].set_ylabel("Hs (m)")
    else:
        axes[0].text(0.5, 0.5, "HTSGW not found", ha="center", va="center")
        axes[0].set_ylabel("Hs (m)")

    axes[0].set_title("GFS-Wave / WAVEWATCH III near MERMAID")

    if "PERPW" in df:
        axes[1].plot(df["valid_time"], df["PERPW"], marker="o")
        axes[1].set_ylabel("Primary period (s)")
    else:
        axes[1].text(0.5, 0.5, "PERPW not found", ha="center", va="center")
        axes[1].set_ylabel("Primary period (s)")

    if "DIRPW" in df:
        axes[2].plot(df["valid_time"], df["DIRPW"], marker="o")
        axes[2].set_ylabel("Primary direction (deg)")
    else:
        axes[2].text(0.5, 0.5, "DIRPW not found", ha="center", va="center")
        axes[2].set_ylabel("Primary direction (deg)")

    axes[2].set_xlabel("Time")

    if event_time is not None:
        event_time = pd.to_datetime(event_time)
        for ax in axes:
            ax.axvline(event_time, linestyle="--", label="MERMAID event")
            ax.legend()

    for ax in axes:
        ax.grid(True)

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def build_wavewatch_point_timeseries(
    center_time,
    lat,
    lon,
    hours_before=24,
    hours_after=24,
    freq="3h",
    product="epacif.0p16",
):
    """
    Use full xarray fields internally, but return a clean point DataFrame
    for time-series plotting.

    This downloads/extracts HTSGW, PERPW, and DIRPW at the nearest grid point.
    """

    center_time = pd.to_datetime(center_time)

    times = pd.date_range(
        center_time - pd.Timedelta(hours=hours_before),
        center_time + pd.Timedelta(hours=hours_after),
        freq=freq,
    )

    variables = {
        "HTSGW": ":HTSGW:",
        "PERPW": ":PERPW:",
        "DIRPW": ":DIRPW:",
    }

    rows = []

    for valid_time in times:
        row = {
            "valid_time": valid_time,
            "lat_requested": lat,
            "lon_requested": lon,
        }

        for output_name, variable_regex in variables.items():
            try:
                ds = gfswave_field(
                    valid_time=valid_time,
                    product=product,
                    variable_regex=variable_regex,
                )

                record = extract_nearest_point_record(
                    ds,
                    lat=lat,
                    lon=lon,
                    output_name=output_name,
                )

                row[output_name] = record[output_name]
                row["lat_model"] = record["lat_model"]
                row["lon_model"] = record["lon_model"]

            except Exception as e:
                print(f"Failed {output_name} at {valid_time}: {e}")
                row[output_name] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def plot_wave_vs_noise(df, noise_col="noise_rms", wave_col="HTSGW"):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df["valid_time"], df[wave_col], label=wave_col)
    ax1.set_ylabel(wave_col)
    ax1.set_xlabel("Time")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(df["valid_time"], df[noise_col], linestyle="--", label=noise_col)
    ax2.set_ylabel(noise_col)

    fig.suptitle(f"{wave_col} vs MERMAID noise")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def plot_wave_noise_scatter(df, wave_col="HTSGW", noise_col="noise_rms"):
    plt.figure(figsize=(6, 5))
    plt.scatter(df[wave_col], df[noise_col], alpha=0.6)
    plt.xlabel(wave_col)
    plt.ylabel(noise_col)
    plt.title(f"{noise_col} vs {wave_col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    event_time = "2022-06-15 15:00"
    lat0 = -14.4512
    lon0 = -179.5054

    # ------------------------------------------------------------
    # 1. Full spatial field plot
    # ------------------------------------------------------------
    field = gfswave_field(
        valid_time=event_time,
        product="global.0p25",
        variable_regex=":HTSGW:",
    )

    plot_wave_height_map(
        field,
        lat0=lat0,
        lon0=lon0,
    )

    # # ------------------------------------------------------------
    # # 2. Optional: extract nearest point from that field
    # # ------------------------------------------------------------
    record = extract_nearest_point_record(
        field,
        lat=lat0,
        lon=lon0,
        output_name="HTSGW",
    )

    print(record)

    # ------------------------------------------------------------
    # 3. Optional: build point time series around the event
    # ------------------------------------------------------------
    # df = build_wavewatch_point_timeseries(
    #     center_time=event_time,
    #     lat=lat0,
    #     lon=lon0,
    #     hours_before=24,
    #     hours_after=24,
    #     freq="3h",
    #     product="epacif.0p16",
    # )

    # print(df)

    # plot_wavewatch_timeseries(
    #     df,
    #     event_time=event_time,
    # )