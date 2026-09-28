import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================
# 1. FILE FINDER
# ==========================================
def find_matching_files(era5land_dir, years, months, cycles):
    import glob

    matched_files = []

    for year in years:
        for month in months:
            mm = f"{int(month):02d}"
            for cycle in cycles:

                pattern = os.path.join(
                    era5land_dir,
                    f"Metcoop_era5land_{year}{mm}_{cycle}.nc"
                )

                files = glob.glob(pattern)
                print(files)

                if files:
                    for f in files:
                        matched_files.append(f)
                        print(f"[FOUND] {os.path.basename(f)}")
                else:
                    print(f"[MISSING] {year}{mm}_{cycle}")

    return sorted(matched_files)


# ==========================================
# 2. SUMMARY STATS
# ==========================================
def calculate_summary_stats(zarr_path):

    ds = xr.open_dataset(zarr_path, engine='zarr', chunks={})

    stats = {'min': [], 'max': [], 'mean': [], 'std': []}
    var_names = []

    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.number):
            d = ds[var]

            stats['min'].append(float(d.min()))
            stats['max'].append(float(d.max()))
            stats['mean'].append(float(d.mean()))
            stats['std'].append(float(d.std()))

            var_names.append(var)

    stats_ds = xr.Dataset(
        {k: (('variable',), v) for k, v in stats.items()},
        coords={'variable': var_names}
    )

    stats_ds.to_zarr(zarr_path, mode='a', zarr_format=2)


# ==========================================
# 3. ZARR SAVING
# ==========================================
def save_to_zarr(zarr_path, ds_era5land):

    ds_to_save = ds_era5land.sortby('valid_time')

    ds_to_save = ds_to_save.chunk({
        'valid_time': 1,
        'latitude': -1,
        'longitude': -1
    })

    zarr_group = os.path.join(zarr_path, '.zgroup')

    if not os.path.exists(zarr_group):
        print(f"Initializing Zarr store at {zarr_path}")
        ds_to_save.to_zarr(zarr_path, mode='w', zarr_format=2)
    else:
        print(f"Appending to Zarr store at {zarr_path}")
        ds_to_save.to_zarr(
            zarr_path,
            mode='a',
            append_dim='valid_time',
            zarr_format=2
        )


# ==========================================
# 4. CROPPING
# ==========================================
def crop_era5land(ds, lat_min, lat_max, lon_min, lon_max, grid_size=128):

    if ds.latitude[0] > ds.latitude[-1]:
        ds_sub = ds.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )
    else:
        ds_sub = ds.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max)
        )

    ds_sub = ds_sub.isel(
        latitude=slice(0, grid_size),
        longitude=slice(0, grid_size)
    )

    print(f"Crop: latitude[0:{grid_size}], longitude[0:{grid_size}]")

    return ds_sub


# ==========================================
# 5. PLOTTING
# ==========================================
def plot_all(ds_era5land, idx, varname='t2m', title_suffix="", save_path=None):

    variable_info = {
        "t2m":    ("2m Temperature", "K", "RdYlBu_r"),
        "d2m":    ("2m Dewpoint Temperature", "K", "RdYlBu_r"),
        "skt":    ("Skin Temperature", "K", "RdYlBu_r"),
        "stl1":   ("Soil Temperature Level 1", "K", "RdYlBu_r"),
        "stl2":   ("Soil Temperature Level 2", "K", "RdYlBu_r"),
        "stl3":   ("Soil Temperature Level 3", "K", "RdYlBu_r"),
        "stl4":   ("Soil Temperature Level 4", "K", "RdYlBu_r"),
        "swvl1":  ("Volumetric Soil Water Layer 1", "m3 m-3", "YlGnBu"),
        "swvl2":  ("Volumetric Soil Water Layer 2", "m3 m-3", "YlGnBu"),
        "swvl3":  ("Volumetric Soil Water Layer 3", "m3 m-3", "YlGnBu"),
        "swvl4":  ("Volumetric Soil Water Layer 4", "m3 m-3", "YlGnBu"),
        "tp":     ("Total Precipitation", "m", "Blues"),
        "ssr":    ("Surface Net Solar Radiation", "J m-2", "YlOrRd"),
        "str":    ("Surface Net Thermal Radiation", "J m-2", "RdBu_r"),
        "slhf":   ("Surface Latent Heat Flux", "J m-2", "PuOr"),
        "sshf":   ("Surface Sensible Heat Flux", "J m-2", "PuOr"),
        "pev":    ("Potential Evaporation", "m", "BrBG"),
        "e":      ("Total Evaporation", "m water equivalent", "BrBG"),
        "sro":    ("Surface Runoff", "m", "Blues"),
        "ssro":   ("Sub-surface Runoff", "m", "Blues"),
        "u10":    ("10m U Wind Component", "m s-1", "RdBu_r"),
        "v10":    ("10m V Wind Component", "m s-1", "RdBu_r"),
        "sp":     ("Surface Pressure", "Pa", "viridis"),
        "lai_hv": ("Leaf Area Index High Vegetation", "m2 m-2", "Greens"),
        "lai_lv": ("Leaf Area Index Low Vegetation", "m2 m-2", "Greens"),
    }

    long_name, units, cmap = variable_info.get(
        varname,
        (varname, "", "viridis")
    )

    data = ds_era5land[varname].isel(**idx)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(10, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    data.plot(
        ax=ax,
        x='longitude',
        y='latitude',
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        cbar_kwargs={
            'shrink': 0.8,
            'label': f"{long_name} ({units})"
        }
    )

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    ax.set_title(f"ERA5-Land {long_name} {title_suffix}")

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)
