import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm

from mermaid_depth.misc.read_tomocat1 import read_tomocat1


def plot_mermaid_location(
    lat_min=None,
    lat_max=None,
    lon_min=None,
    lon_max=None,
    downsample=10,
    grid_labels=False,
):
    # ----------------------------
    # Read MERMAID locations
    # ----------------------------
    data = read_tomocat1("./tomocat1.txt")
    mer_lats = np.asarray(data["stla"], dtype=float)
    mer_lons = np.asarray(data["stlo"], dtype=float)

    # ----------------------------
    # Read GEBCO
    # ----------------------------
    ds = xr.open_dataset("./GEBCO_2025.nc")
    lat = ds["lat"].values
    lon = ds["lon"].values
    elevation = ds["elevation"]

    # ----------------------------
    # Choose longitude center
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

    # Shift longitudes so the desired box is continuous
    lon_shifted = ((lon - central_lon + 180) % 360) - 180 + central_lon
    mer_lons_shifted = ((mer_lons - central_lon + 180) % 360) - 180 + central_lon

    # ----------------------------
    # Subset indices first
    # ----------------------------
    if (
        lat_min is not None
        and lat_max is not None
        and lon_min is not None
        and lon_max is not None
    ):
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx = np.where((lon_shifted >= lon_min) & (lon_shifted <= lon_max))[0]

        mer_keep = (
            (mer_lats >= lat_min)
            & (mer_lats <= lat_max)
            & (mer_lons_shifted >= lon_min)
            & (mer_lons_shifted <= lon_max)
        )
        mer_lats = mer_lats[mer_keep]
        mer_lons_shifted = mer_lons_shifted[mer_keep]
    else:
        lat_idx = np.arange(len(lat))
        lon_idx = np.arange(len(lon))

    # Sort longitude indices so pcolormesh gets increasing x
    lon_idx = lon_idx[np.argsort(lon_shifted[lon_idx])]

    # Downsample AFTER subsetting
    lat_idx = lat_idx[::downsample]
    lon_idx = lon_idx[::downsample]

    # Extract only the needed subset
    lat_plot = lat[lat_idx]
    lon_plot = lon_shifted[lon_idx]
    elev_plot = elevation.isel(lat=lat_idx, lon=lon_idx).values

    # ----------------------------
    # Plot
    # ----------------------------
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

    ax.scatter(
        mer_lons_shifted,
        mer_lats,
        transform=ccrs.PlateCarree(),
        color="red",
        edgecolor="black",
        s=50,
        marker="v",
        zorder=5,
    )

    if (
        lat_min is not None
        and lat_max is not None
        and lon_min is not None
        and lon_max is not None
    ):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    ax.coastlines(resolution="110m")

    gl = ax.gridlines(draw_labels=grid_labels)
    if grid_labels:
        gl.top_labels = False
        gl.right_labels = False

    plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        label="GEBCO Elevation [m]",
    )
    plt.title("MERMAID Instrument Locations")
    plt.show()


if __name__ == "__main__":
    plot_mermaid_location(
        lat_min=-40,
        lat_max=10,
        lon_min=-220,
        lon_max=-100,
        downsample=50,
        grid_labels=False,
    )