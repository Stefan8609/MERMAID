import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

from mermaid_depth.misc.read_tomocat1 import read_tomocat1


def parse_tomocat_time(s):
    s = str(s)

    date_part, clock_part = s.split("T")
    hh, mm, sec_plus_b = clock_part.split(":")

    # First two characters are the SAC reference seconds.
    sec = sec_plus_b[:2]

    # Everything after the first two characters is the B-header offset.
    b_string = sec_plus_b[2:]

    if b_string == "":
        b_offset = 0.0
    else:
        b_offset = float(b_string)

    ref_time = pd.Timestamp(f"{date_part}T{hh}:{mm}:{sec}")

    return ref_time + pd.to_timedelta(b_offset, unit="s")


def plot_mermaid_location_intervals(
    start_time,
    n_intervals=8,
    interval_hours=3,
    lat_min=None,
    lat_max=None,
    lon_min=None,
    lon_max=None,
    downsample=50,
    save_dir=None,
):
    # ----------------------------
    # Read MERMAID data
    # ----------------------------
    data = read_tomocat1("./tomocat1.txt")

    mer_lats = np.asarray(data["stla"], dtype=float)
    mer_lons = np.asarray(data["stlo"], dtype=float)
    time_strings = np.asarray(data["seismogram_time"], dtype=str)

    # Parse TOMOCAT-style seismogram times.
    times = pd.DatetimeIndex([parse_tomocat_time(s) for s in time_strings])

    # Parse starting time using the same convention.
    start_time = parse_tomocat_time(start_time)
    interval = pd.Timedelta(hours=interval_hours)

    # ----------------------------
    # Load GEBCO
    # ----------------------------
    ds = xr.open_dataset("./GEBCO_2025.nc")
    lat = ds["lat"].values
    lon = ds["lon"].values
    elevation = ds["elevation"]

    # ----------------------------
    # Set longitude center
    # ----------------------------
    if (
        lat_min is not None
        and lat_max is not None
        and lon_min is not None
        and lon_max is not None
    ):
        central_lon = 0.5 * (lon_min + lon_max)
    else:
        central_lon = 0.0

    # Shift longitudes so boxes like lon_min=-220, lon_max=-150 work.
    lon_shifted = ((lon - central_lon + 180) % 360) - 180 + central_lon
    mer_lons_shifted = ((mer_lons - central_lon + 180) % 360) - 180 + central_lon

    # ----------------------------
    # Subset GEBCO and MERMAIDs spatially
    # ----------------------------
    if (
        lat_min is not None
        and lat_max is not None
        and lon_min is not None
        and lon_max is not None
    ):
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx = np.where((lon_shifted >= lon_min) & (lon_shifted <= lon_max))[0]

        space_keep = (
            (mer_lats >= lat_min)
            & (mer_lats <= lat_max)
            & (mer_lons_shifted >= lon_min)
            & (mer_lons_shifted <= lon_max)
        )

        mer_lats = mer_lats[space_keep]
        mer_lons_shifted = mer_lons_shifted[space_keep]
        times = times[space_keep]

    else:
        lat_idx = np.arange(len(lat))
        lon_idx = np.arange(len(lon))

    # Sort longitudes for pcolormesh.
    lon_idx = lon_idx[np.argsort(lon_shifted[lon_idx])]

    # Downsample GEBCO after subsetting.
    if downsample is not None and downsample > 1:
        lat_idx = lat_idx[::downsample]
        lon_idx = lon_idx[::downsample]

    lat_plot = lat[lat_idx]
    lon_plot = lon_shifted[lon_idx]
    elev_plot = elevation.isel(lat=lat_idx, lon=lon_idx).values

    # Optional output directory.
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Plot each time interval
    # ----------------------------
    for i in range(n_intervals):
        t0 = start_time + i * interval
        t1 = t0 + interval

        time_keep = (times >= t0) & (times < t1)

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mercator(central_longitude=central_lon))

        im = ax.pcolormesh(
            lon_plot,
            lat_plot,
            elev_plot,
            transform=ccrs.PlateCarree(),
            cmap="terrain",
            norm=TwoSlopeNorm(vmin=-6000, vcenter=0, vmax=2000),
            shading="auto",
            rasterized=True,
        )

        # All spatially valid MERMAID points in gray.
        ax.scatter(
            mer_lons_shifted,
            mer_lats,
            transform=ccrs.PlateCarree(),
            color="0.7",
            edgecolor="none",
            s=20,
            marker="v",
            zorder=4,
            label="All points in box",
        )

        # Current interval in red.
        ax.scatter(
            mer_lons_shifted[time_keep],
            mer_lats[time_keep],
            transform=ccrs.PlateCarree(),
            color="red",
            edgecolor="black",
            s=60,
            marker="v",
            zorder=5,
            label="Current interval",
        )

        if (
            lat_min is not None
            and lat_max is not None
            and lon_min is not None
            and lon_max is not None
        ):
            ax.set_extent(
                [lon_min, lon_max, lat_min, lat_max],
                crs=ccrs.PlateCarree(),
            )
        else:
            ax.set_global()

        ax.coastlines(resolution="110m")
        ax.gridlines(draw_labels=False)

        plt.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            pad=0.08,
            label="GEBCO Elevation [m]",
        )

        ax.legend(loc="lower left")

        plt.title(
            f"MERMAID Locations\n"
            f"{t0.strftime('%Y-%m-%d %H:%M:%S')} to "
            f"{t1.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{np.sum(time_keep)} points"
        )

        fig.tight_layout()

        if save_dir is not None:
            outfile = save_dir / f"mermaids_{t0.strftime('%Y%m%dT%H%M%S')}.png"
            fig.savefig(outfile, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    plot_mermaid_location_intervals(
        start_time="2018-08-27T00:00:000.00",
        n_intervals=12,
        interval_hours=168,
        lat_min=-50,
        lat_max=20,
        lon_min=-260,
        lon_max=-100,
        downsample=50,
        # save_dir="mermaid_3hr_frames",
    )