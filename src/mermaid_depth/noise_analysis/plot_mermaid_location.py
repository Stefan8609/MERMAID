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
):
    # Read MERMAID locations
    data = read_tomocat1("./tomocat1.txt")
    latitudes = np.asarray(data["stla"], dtype=float)
    longitudes = np.asarray(data["stlo"], dtype=float)

    # Wrap longitudes to [-180, 180]
    longitudes = ((longitudes + 180) % 360) - 180

    # Load GEBCO
    ds = xr.open_dataset("./GEBCO_2025.nc")
    elevation = ds["elevation"]
    lons = ((ds["lon"].values + 180) % 360) - 180

    # Put wrapped longitudes into the data array and sort
    elevation = elevation.assign_coords(lon=lons).sortby("lon")

    # If a map box is given, subset GEBCO and MERMAIDs first
    if (
        lat_min is not None
        and lat_max is not None
        and lon_min is not None
        and lon_max is not None
    ):
        elevation = elevation.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )

        keep = (
            (latitudes >= lat_min)
            & (latitudes <= lat_max)
            & (longitudes >= lon_min)
            & (longitudes <= lon_max)
        )
        latitudes = latitudes[keep]
        longitudes = longitudes[keep]

    # Downsample GEBCO for speed
    if downsample is not None and downsample > 1:
        elevation = elevation.isel(
            lat=slice(None, None, downsample),
            lon=slice(None, None, downsample),
        )

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.Mercator())

    # Plot GEBCO
    im = ax.pcolormesh(
        elevation["lon"].values,
        elevation["lat"].values,
        elevation.values,
        transform=ccrs.PlateCarree(),
        cmap="terrain",
        norm=TwoSlopeNorm(vmin=-6000, vcenter=0, vmax=2000),
        shading="auto",
        rasterized=True,
    )

    # Plot MERMAIDs
    ax.scatter(
        longitudes,
        latitudes,
        transform=ccrs.PlateCarree(),
        color="red",
        edgecolor="black",
        s=50,
        marker="v",
        zorder=5,
    )

    # Set map extent
    if (
        lat_min is not None
        and lat_max is not None
        and lon_min is not None
        and lon_max is not None
    ):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    # Lower-resolution coastlines are faster
    ax.coastlines(resolution="110m")
    ax.gridlines(draw_labels=True)

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
        lon_min=-200,
        lon_max=-120,
        downsample=50,
    )