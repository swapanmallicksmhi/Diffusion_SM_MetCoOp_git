#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from matplotlib.colors import BoundaryNorm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot selected variables from ERA5-Land Zarr dataset"
    )

    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)

    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)

    return parser.parse_args()


def get_colormap(var):
    v = var.lower()

    if v in ["t2m", "d2m", "skt", "stl1", "stl2", "stl3", "stl4"]:
        levels = np.arange(242, 312, 2)
        cmap = cmocean.cm.thermal.resampled(len(levels) - 1)
        norm = BoundaryNorm(levels, cmap.N)
        return cmap, norm, levels
    elif v in ["swvl1", "swvl2", "swvl3", "swvl4"]:
        return "YlGnBu", None, None
    elif v in ["tp", "sro", "ssro"]:
        return "Blues", None, None
    elif v in ["ssr"]:
        return "YlOrRd", None, None
    elif v in ["str"]:
        return "RdBu_r", None, None
    elif v in ["slhf", "sshf"]:
        return "PuOr", None, None
    elif v in ["pev", "e"]:
        return "BrBG", None, None
    elif v in ["u10", "v10"]:
        return "coolwarm", None, None
    elif v in ["sp"]:
        return "cividis", None, None
    elif v in ["lai_hv", "lai_lv"]:
        return "Greens", None, None
    else:
        return "viridis", None, None


def get_label(var):
    labels = {
        "t2m": "2 m temperature (K)",
        "d2m": "2 m dewpoint temperature (K)",
        "skt": "Skin temperature (K)",
        "stl1": "Soil temperature level 1 (K)",
        "stl2": "Soil temperature level 2 (K)",
        "stl3": "Soil temperature level 3 (K)",
        "stl4": "Soil temperature level 4 (K)",
        "swvl1": "Volumetric soil water layer 1 (m3 m-3)",
        "swvl2": "Volumetric soil water layer 2 (m3 m-3)",
        "swvl3": "Volumetric soil water layer 3 (m3 m-3)",
        "swvl4": "Volumetric soil water layer 4 (m3 m-3)",
        "tp": "Total precipitation (m)",
        "ssr": "Surface net solar radiation (J m-2)",
        "str": "Surface net thermal radiation (J m-2)",
        "slhf": "Surface latent heat flux (J m-2)",
        "sshf": "Surface sensible heat flux (J m-2)",
        "pev": "Potential evaporation (m)",
        "e": "Total evaporation (m water equivalent)",
        "sro": "Surface runoff (m)",
        "ssro": "Sub-surface runoff (m)",
        "u10": "10 m U wind component (m s-1)",
        "v10": "10 m V wind component (m s-1)",
        "sp": "Surface pressure (Pa)",
        "lai_hv": "Leaf area index high vegetation (m2 m-2)",
        "lai_lv": "Leaf area index low vegetation (m2 m-2)",
    }

    return labels.get(var, var)


def crop_latlon(field, lat_min, lat_max, lon_min, lon_max):
    lat0 = float(field.latitude.values[0])
    lat1 = float(field.latitude.values[-1])

    if lat0 > lat1:
        field = field.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max),
        )
    else:
        field = field.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )

    return field


def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Opening Zarr dataset")
    print(args.zarr_path)
    print("=" * 80)

    ds = xr.open_dataset(
        args.zarr_path,
        engine="zarr",
        chunks={"valid_time": 1},
    )

    print(ds)

    if "valid_time" not in ds.coords:
        raise ValueError("valid_time coordinate not found in Zarr dataset")

    ds["valid_time"] = pd.to_datetime(ds["valid_time"].values)
    ds = ds.sortby("valid_time")

    print("=" * 80)
    print("Available time range")
    print("First time:", ds.valid_time.values[0])
    print("Last time :", ds.valid_time.values[-1])

    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date)

    ds = ds.sel(valid_time=slice(start, end))

    if ds.sizes["valid_time"] == 0:
        raise ValueError(
            f"No data found between {args.start_date} and {args.end_date}"
        )

    print("=" * 80)
    print("Selected time range")
    print("First selected time:", ds.valid_time.values[0])
    print("Last selected time :", ds.valid_time.values[-1])
    print("Number of selected times:", ds.sizes["valid_time"])

    variables = [
        "t2m",
        "d2m",
        "skt",
        "stl1",
        "stl2",
        "stl3",
        "stl4",
        "swvl1",
        "swvl2",
        "swvl3",
        "swvl4",
        "tp",
        "ssr",
        "str",
        "slhf",
        "sshf",
        "pev",
        "e",
        "sro",
        "ssro",
        "u10",
        "v10",
        "sp",
        "lai_hv",
        "lai_lv",
    ]

    variables = [v for v in variables if v in ds.data_vars]

    print("=" * 80)
    print("Variables to plot")
    for v in variables:
        print(v)

    if not variables:
        raise ValueError("No expected ERA5-Land variables found in Zarr")

    times = ds.valid_time.values

    for var in variables:

        print("=" * 80)
        print("Processing variable:", var)

        var_dir = os.path.join(args.output_dir, var)
        os.makedirs(var_dir, exist_ok=True)

        da = ds[var]

        for t in range(len(times)):

            field = da.isel(valid_time=t)

            if "number" in field.dims:
                field = field.mean("number")

            field = crop_latlon(
                field,
                args.lat_min,
                args.lat_max,
                args.lon_min,
                args.lon_max,
            )

            data = field.load()

            if np.isnan(data.values).all():
                print("Skipping empty field:", var, times[t])
                continue

            vmin = np.nanpercentile(data.values, 2)
            vmax = np.nanpercentile(data.values, 98)

            if not np.isfinite(vmin) or not np.isfinite(vmax):
                print("Skipping invalid field:", var, times[t])
                continue

            if vmin == vmax:
                vmin = float(np.nanmin(data.values))
                vmax = float(np.nanmax(data.values))

            cmap, norm, levels = get_colormap(var)

            tval = str(times[t])
            tstr = (
                tval[:16]
                .replace(":", "")
                .replace("-", "")
                .replace("T", "_")
                .replace(" ", "_")
            )

            save_path = os.path.join(var_dir, f"{var}_{tstr}.png")

            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())

            if norm is not None:
                mesh = ax.pcolormesh(
                    data["longitude"],
                    data["latitude"],
                    data,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    cmap=cmap,
                    norm=norm,
                )
            else:
                mesh = ax.pcolormesh(
                    data["longitude"],
                    data["latitude"],
                    data,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

            ax.set_extent(
                [args.lon_min, args.lon_max, args.lat_min, args.lat_max],
                crs=ccrs.PlateCarree(),
            )

            ax.coastlines(resolution="10m", linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)

            gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False

            ax.set_title(f"ERA5-Land {get_label(var)}\n{tval}")

            if levels is not None:
                cbar = plt.colorbar(
                    mesh,
                    ax=ax,
                    shrink=0.75,
                    pad=0.03,
                    boundaries=levels,
                    ticks=levels,
                )
            else:
                cbar = plt.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03)

            cbar.set_label(get_label(var))

            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            print("Saved:", save_path)

    ds.close()

    print("=" * 80)
    print("Done. All plots saved in:")
    print(args.output_dir)


if __name__ == "__main__":
    main()
