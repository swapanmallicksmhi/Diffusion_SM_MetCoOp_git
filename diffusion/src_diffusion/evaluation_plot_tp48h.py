#!/usr/bin/env python3
"""
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
import torch
from datetime import datetime, timedelta


class PlotGenerator:
    """Generates geographic plots with longitude/latitude information"""
    
    # Fixed grid size for all plots
    GRID_SIZE = (1200, 784)
    
    def __init__(self, zarr_path, output_dir, shapefile=None):
        """
        Initialize with Zarr path to load longitude/latitude
        
        Args:
            zarr_path: Path to Zarr store with latitude/longitude coordinates
            output_dir: Directory to save plots
        """
        self.zarr_path = zarr_path
        self.output_dir = output_dir
        self.shapefile = shapefile
        self.valid_mask = None
        self.boundary = None
        os.makedirs(output_dir, exist_ok=True)
        
        # Load latitude and longitude from Zarr
        self._load_coordinates()
        
        # Load state/administrative boundary if provided
        if self.shapefile:
            self._load_boundary()

        # Store datetime information
        self.conditioning_datetime = None
    
    def set_conditioning_datetime(self, datetime_obj):
        """Set the conditioning datetime for plot titles"""
        self.conditioning_datetime = datetime_obj


    def set_valid_mask(self, valid_mask):
        """Set valid geographical mask. Invalid/ocean points will be white."""
        mask = np.asarray(valid_mask, dtype=bool)

        expected_shape = (
            len(self.lat),
            len(self.lon)
        )

        if mask.shape != expected_shape:
            raise ValueError(
                f"Valid-mask shape {mask.shape} does not match "
                f"geographical grid {expected_shape}"
            )

        self.valid_mask = mask

        print(
            f"Plot valid mask set: "
            f"{int(mask.sum())}/{mask.size} valid grid points"
        )

    def _load_boundary(self):
        """Load India state/administrative boundary shapefile."""
        print(f"Loading boundary shapefile: {self.shapefile}")

        boundary = gpd.read_file(self.shapefile)

        if boundary.crs is None:
            boundary = boundary.set_crs(epsg=4326)
        else:
            boundary = boundary.to_crs(epsg=4326)

        self.boundary = boundary

        print(
            f"Boundary features loaded: {len(self.boundary)}"
        )

    def _mask_plot_data(self, data_2d):
        """Mask invalid/ocean grid points with NaN."""
        data_2d = np.asarray(data_2d, dtype=np.float32)

        if self.valid_mask is not None:
            if data_2d.shape != self.valid_mask.shape:
                raise ValueError(
                    f"Plot-data shape {data_2d.shape} does not match "
                    f"valid-mask shape {self.valid_mask.shape}"
                )

            data_2d = data_2d.copy()
            data_2d[~self.valid_mask] = np.nan

        return data_2d

    def _get_plot_cmap(self, cmap):
        """Return a colormap whose NaN/bad values are white."""
        if isinstance(cmap, str):
            plot_cmap = plt.get_cmap(cmap).copy()
        else:
            plot_cmap = cmap.copy() if hasattr(cmap, "copy") else cmap

        if hasattr(plot_cmap, "set_bad"):
            plot_cmap.set_bad(color="white")

        return plot_cmap

    def _add_boundary(self, ax, lon_min, lon_max, lat_min, lat_max):
        """Add India state/administrative boundary to a Cartopy axis."""
        if self.boundary is None:
            return

        boundary_subset = self.boundary.cx[
            lon_min:lon_max,
            lat_min:lat_max
        ]

        if not boundary_subset.empty:
            boundary_subset.boundary.plot(
                ax=ax,
                edgecolor="black",
                linewidth=1.2,
                transform=ccrs.PlateCarree(),
                zorder=5
            )
    
    def _load_coordinates(self):
        """Load latitude, longitude and training-variable ranges from Zarr store"""
        print(f"\n{'='*60}")
        print("Loading geographic coordinates from Zarr")
        print(f"{'='*60}")
        
        ds = xr.open_dataset(self.zarr_path, engine="zarr")
        
        # Extract latitude and longitude
        if 'latitude' in ds:
            self.lat = ds['latitude'].values
            print(
                f"Latitude shape: {self.lat.shape}, "
                f"range: [{self.lat.min():.2f}, {self.lat.max():.2f}]"
            )
        else:
            raise ValueError("No 'latitude' variable found in Zarr store")
        
        if 'longitude' in ds:
            self.lon = ds['longitude'].values
            print(
                f"Longitude shape: {self.lon.shape}, "
                f"range: [{self.lon.min():.2f}, {self.lon.max():.2f}]"
            )
        else:
            raise ValueError("No 'longitude' variable found in Zarr store")
        
        # Handle longitude wrapping (convert 0-360 to -180 to 180 if needed)
        if self.lon.max() > 180:
            print("Converting longitude from 0-360 to -180-180 range")
            self.lon = np.where(
                self.lon > 180,
                self.lon - 360,
                self.lon
            )
            print(
                f"Adjusted longitude range: "
                f"[{self.lon.min():.2f}, {self.lon.max():.2f}]"
            )

        # Load original training-data ranges from Zarr
        self.training_ranges = {}

        if (
            'variable' in ds
            and 'min' in ds
            and 'max' in ds
        ):
            variable_names = ds['variable'].values
            min_values = ds['min'].values
            max_values = ds['max'].values

            print(f"\n{'='*60}")
            print("Loading original variable ranges from training Zarr")
            print(f"{'='*60}")

            for i, var in enumerate(variable_names):

                if isinstance(var, bytes):
                    var_name = var.decode("utf-8")
                else:
                    var_name = str(var)

                var_min = float(min_values[i])
                var_max = float(max_values[i])

                self.training_ranges[var_name] = (
                    var_min,
                    var_max
                )

                print(
                    f"{var_name:10s}: "
                    f"min = {var_min:.6g}, "
                    f"max = {var_max:.6g}"
                )

        else:
            print(
                "Warning: variable/min/max metadata not found. "
                "Using default plotting scales."
            )
        
        ds.close()
    
    def _get_variable_info(self, var_name):
        """Get colormap, fixed levels and label using ERA5-Land plotting style."""
        v = var_name.lower()

        labels = {
            "t2m": "2 m temperature (K)",
            "d2m": "2 m dewpoint temperature (K)",
            "skt": "Skin temperature (K)",
            "stl1": "Soil temperature level 1 (K)",
            "swvl1": "Volumetric soil water layer 1 (%)",
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

        label = labels.get(var_name, var_name)

        # Fixed temperature scale
        if v in ["t2m", "d2m", "skt", "stl1"]:
            levels = np.arange(252, 322, 2)
            cmap = plt.get_cmap("jet", len(levels) - 1)
            norm = BoundaryNorm(levels, cmap.N)
            return levels, cmap, norm, label

        # Fixed soil-moisture scale, plotted as percentage
        elif v == "swvl1":
            levels = np.arange(5.0, 65.0, 5.0)

            moist_colors = [
                "#8c510a",
                "#bf812d",
                "#dfc27d",
                "#f6e8a6",
                "#ffffbf",
                "#c7eae5",
                "#80cdc1",
                "#3598c5",
                "#0165b8",
                "#08306b",
                "#041f4a",
            ]

            cmap = ListedColormap(moist_colors)
            norm = BoundaryNorm(levels, cmap.N)
            return levels, cmap, norm, label

        # Variable-specific colormaps
        if v in ["tp", "sro", "ssro"]:
            cmap = "Blues"
        elif v in ["ssr"]:
            cmap = "YlOrRd"
        elif v in ["str"]:
            cmap = "RdBu_r"
        elif v in ["slhf", "sshf"]:
            cmap = "PuOr"
        elif v in ["pev", "e"]:
            cmap = "BrBG"
        elif v in ["u10", "v10"]:
            cmap = "coolwarm"
        elif v in ["sp"]:
            cmap = "cividis"
        elif v in ["lai_hv", "lai_lv"]:
            cmap = "Greens"
        else:
            cmap = "viridis"

        return None, cmap, None, label

    def _prepare_plot_field(self, data_2d, var_name):
        """
        Prepare a field for plotting only.

        swvl1 is converted from m3 m-3 to percent for plotting.
        Model output, NPY files and verification metrics remain unchanged.
        """
        data_plot = self._mask_plot_data(data_2d)

        if var_name is not None and var_name.lower() == "swvl1":
            data_plot = data_plot * 100.0

        return data_plot

    def _get_dynamic_limits(self, data_2d):
        """Return the 2nd and 98th percentile plotting limits."""
        finite = np.isfinite(data_2d)

        if not np.any(finite):
            return None, None

        vmin = np.nanpercentile(data_2d, 2)
        vmax = np.nanpercentile(data_2d, 98)

        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None, None

        if vmin == vmax:
            vmin = float(np.nanmin(data_2d))
            vmax = float(np.nanmax(data_2d))

        return float(vmin), float(vmax)

    def _format_datetime(self, dt):
        """Format datetime for plot titles"""
        if isinstance(dt, str):
            return dt
        elif isinstance(dt, datetime):
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        elif hasattr(dt, 'strftime'):
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            return str(dt)
    
    def _prepare_data(self, data_tensor, var_name):
        """
        Prepare tensor data for plotting
        
        Args:
            data_tensor: PyTorch tensor of shape [C, H, W] or [H, W]
            var_name: Variable name for identifying mean/std
            
        Returns:
            numpy array ready for plotting
        """
        # Convert to numpy and ensure 2D
        if isinstance(data_tensor, torch.Tensor):
            data_np = data_tensor.cpu().detach().numpy()
        else:
            data_np = data_tensor
        
        # If multi-channel, select appropriate channel
        if data_np.ndim == 3:
            if "std" in var_name.lower() and data_np.shape[0] > 1:
                # Assume last channel is std if multiple
                data_np = data_np[-1]
            else:
                # Take first channel for mean
                data_np = data_np[0]
        
        return data_np
    
    def plot_mean_and_std_separate(
        self,
        samples_denorm,
        sample_idx,
        var_names,
        conditioning_time=None,
        forecast_valid_time=None,
        forecast_lead_hours=0,
        lon_min=None,
        lon_max=None,
        lat_min=None,
        lat_max=None,
        output_subdir=None
    ):
        """
        Generate separate plots for mean and standard deviation with datetime

        Args:
            samples_denorm: Denormalized samples tensor [num_samples, channels, H, W]
            sample_idx: Index of sample to plot
            var_names: List of variable names
            conditioning_time: Initialization datetime of conditioning data
            forecast_valid_time: Forecast valid datetime
            forecast_lead_hours: Forecast lead time in hours
            lon_min, lon_max, lat_min, lat_max: Optional domain subset
            output_subdir: Optional subdirectory to save plots in
        """
        # Set default domain if not provided
        if lon_min is None:
            lon_min = float(self.lon.min())
        if lon_max is None:
            lon_max = float(self.lon.max())
        if lat_min is None:
            lat_min = float(self.lat.min())
        if lat_max is None:
            lat_max = float(self.lat.max())

        # Determine output directory
        if output_subdir:
            save_dir = output_subdir
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir

        # Format initialization and forecast-valid times for title
        time_str = self._format_datetime(conditioning_time) if conditioning_time else "Unknown"
        valid_time_str = (
            self._format_datetime(forecast_valid_time)
            if forecast_valid_time is not None
            else "Unknown"
        )

        safe_time = time_str.replace(" ", "_").replace(":", "").replace("-", "")
        safe_valid_time = (
            valid_time_str.replace(" ", "_").replace(":", "").replace("-", "")
        )

        # Get the sample
        sample = samples_denorm[sample_idx]

        # Create separate plots for each channel
        for c, var_name in enumerate(var_names):
            if c >= sample.shape[0]:
                break

            # Prepare data
            data_2d = sample[c].cpu().detach().numpy()

            # Get plotting parameters
            levels, cmap, norm, label = self._get_variable_info(var_name)

            # Create filename with datetime
            filename = (
                f"sample_{sample_idx:03d}_{var_name}_"
                f"init_{safe_time}_valid_{safe_valid_time}_"
                f"lead_{int(forecast_lead_hours):03d}h.png"
            )
            save_path = os.path.join(save_dir, filename)

            # Create title with datetime
            title = (
                f"Sample {sample_idx+1} - {label}\n"
                f"Initialization: {time_str}\n"
                f"Valid: {valid_time_str}\n"
                f"Lead: +{int(forecast_lead_hours)} h"
            )

            # Create plot
            self._create_geographic_plot(
                data_2d, 
                lon_min, lon_max, lat_min, lat_max,
                cmap, norm, levels,
                title=title,
                filename=save_path,
                var_name=var_name
            )

            print(f"  Saved: {filename}")

    def plot_all_samples_grid(
        self,
        samples_denorm,
        var_names,
        conditioning_time=None,
        forecast_valid_time=None,
        forecast_lead_hours=0,
        lon_min=None,
        lon_max=None,
        lat_min=None,
        lat_max=None,
        output_subdir=None
    ):
        """
        Create a grid plot of all samples for each variable with datetime

        Args:
            samples_denorm: Denormalized samples tensor [num_samples, channels, H, W]
            var_names: List of variable names
            conditioning_time: Initialization datetime of conditioning data
            forecast_valid_time: Forecast valid datetime
            forecast_lead_hours: Forecast lead time in hours
            lon_min, lon_max, lat_min, lat_max: Optional domain subset
            output_subdir: Optional subdirectory to save plots in
        """
        num_samples = samples_denorm.shape[0]
        num_channels = samples_denorm.shape[1]

        # Determine output directory
        if output_subdir:
            save_dir = output_subdir
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir

        # Format initialization and forecast-valid times
        time_str = self._format_datetime(conditioning_time) if conditioning_time else "Unknown"
        valid_time_str = (
            self._format_datetime(forecast_valid_time)
            if forecast_valid_time is not None
            else "Unknown"
        )

        safe_time = time_str.replace(" ", "_").replace(":", "").replace("-", "")
        safe_valid_time = (
            valid_time_str.replace(" ", "_").replace(":", "").replace("-", "")
        )

        # Set default domain
        if lon_min is None:
            lon_min = float(self.lon.min())
        if lon_max is None:
            lon_max = float(self.lon.max())
        if lat_min is None:
            lat_min = float(self.lat.min())
        if lat_max is None:
            lat_max = float(self.lat.max())

        # Create a grid plot for each variable
        for c in range(min(num_channels, len(var_names))):
            var_name = var_names[c]

            # Get plotting parameters
            levels, cmap, norm, label = self._get_variable_info(var_name)
            plot_cmap = self._get_plot_cmap(cmap)

            # Create figure with subplots
            cols = min(4, num_samples)
            rows = (num_samples + cols - 1) // cols

            fig = plt.figure(figsize=(5*cols, 4*rows))
            fig.suptitle(
                f"{var_name} - All Generated Samples\n"
                f"Initialization: {time_str} | "
                f"Valid: {valid_time_str} | "
                f"Lead: +{int(forecast_lead_hours)} h",
                fontsize=16
            )

            for i in range(num_samples):
                ax = fig.add_subplot(
                    rows,
                    cols,
                    i+1,
                    projection=ccrs.PlateCarree()
                )

                # Get data
                data_2d = samples_denorm[i, c].cpu().detach().numpy()
                data_2d = self._prepare_plot_field(
                    data_2d,
                    var_name
                )

                if norm is None:
                    vmin, vmax = self._get_dynamic_limits(data_2d)
                else:
                    vmin = None
                    vmax = None

                # Create mesh
                if norm is not None:
                    mesh = ax.pcolormesh(
                        self.lon,
                        self.lat,
                        data_2d,
                        transform=ccrs.PlateCarree(),
                        shading="auto",
                        cmap=plot_cmap,
                        norm=norm,
                        zorder=1,
                    )
                else:
                    mesh = ax.pcolormesh(
                        self.lon,
                        self.lat,
                        data_2d,
                        transform=ccrs.PlateCarree(),
                        shading="auto",
                        cmap=plot_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        zorder=1,
                    )

                # Set extent
                ax.set_extent(
                    [lon_min, lon_max, lat_min, lat_max],
                    crs=ccrs.PlateCarree()
                )

                # Add features
                ax.coastlines(
                    resolution="10m",
                    linewidth=0.8,
                    zorder=3
                )
                #ax.add_feature(
                #    cfeature.BORDERS,
                #    linewidth=0.5,
                #    zorder=3
                #)

                self._add_boundary(
                    ax,
                    lon_min,
                    lon_max,
                    lat_min,
                    lat_max
                )

                # Add title with datetime
                ax.set_title(f"Sample {i+1}")

                # Add gridlines
                gl = ax.gridlines(
                    draw_labels=False,
                    linestyle='--',
                    linewidth=0.3
                )

            # Add colorbar
            if levels is not None:
                cbar = fig.colorbar(
                    mesh,
                    ax=fig.axes,
                    shrink=0.75,
                    pad=0.03,
                    boundaries=levels,
                    ticks=levels[::2],
                )
            else:
                cbar = fig.colorbar(
                    mesh,
                    ax=fig.axes,
                    shrink=0.75,
                    pad=0.03
                )

            cbar.set_label(label)

            plt.tight_layout()

            # Save with datetime in filename
            filename = (
                f"all_samples_{var_name}_grid_"
                f"init_{safe_time}_valid_{safe_valid_time}_"
                f"lead_{int(forecast_lead_hours):03d}h.png"
            )
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved grid plot: {filename}")

    def _create_geographic_plot(
        self,
        data_2d,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        cmap,
        norm,
        levels,
        title,
        filename,
        var_name=None
    ):
        """
        Create a single geographic plot with fixed grid size

        Args:
            data_2d: 2D numpy array of data
            lon_min, lon_max, lat_min, lat_max: Domain boundaries
            cmap: Colormap
            norm: Normalization
            levels: Colorbar levels
            title: Plot title (includes datetime)
            filename: Output filename
        """
        data_2d = self._prepare_plot_field(
            data_2d,
            var_name
        )

        plot_cmap = self._get_plot_cmap(cmap)

        if norm is None:
            vmin, vmax = self._get_dynamic_limits(data_2d)
        else:
            vmin = None
            vmax = None

        # Create figure with fixed grid size
        fig = plt.figure(
            figsize=(
                self.GRID_SIZE[0]/100,
                self.GRID_SIZE[1]/100
            ),
            dpi=100
        )

        ax = plt.axes(
            projection=ccrs.PlateCarree()
        )

        # Plot data
        if norm is not None:
            mesh = ax.pcolormesh(
                self.lon,
                self.lat,
                data_2d,
                transform=ccrs.PlateCarree(),
                shading="auto",
                cmap=plot_cmap,
                norm=norm,
                zorder=1,
            )
        else:
            mesh = ax.pcolormesh(
                self.lon,
                self.lat,
                data_2d,
                transform=ccrs.PlateCarree(),
                shading="auto",
                cmap=plot_cmap,
                vmin=vmin,
                vmax=vmax,
                zorder=1,
            )

        # Set extent
        ax.set_extent(
            [lon_min, lon_max, lat_min, lat_max],
            crs=ccrs.PlateCarree()
        )

        # Add geographic features
        ax.coastlines(
            resolution="10m",
            linewidth=0.8,
            color='black',
            zorder=3
        )

        #ax.add_feature(
        #    cfeature.BORDERS,
        #    linewidth=0.5,
        #    edgecolor='black',
        #    zorder=3
        #)

        self._add_boundary(
            ax,
            lon_min,
            lon_max,
            lat_min,
            lat_max
        )

        # Add gridlines with labels
        gl = ax.gridlines(
            draw_labels=True,
            crs=ccrs.PlateCarree(),
            linestyle='--',
            linewidth=0.3
        )

        gl.top_labels = False
        gl.right_labels = False

        # Add title with datetime
        ax.set_title(
            title,
            fontsize=12
        )

        # Add colorbar
        if levels is not None:
            cbar = plt.colorbar(
                mesh,
                ax=ax,
                shrink=0.75,
                pad=0.03,
                boundaries=levels,
                ticks=levels[::2],
            )
        else:
            cbar = plt.colorbar(
                mesh,
                ax=ax,
                shrink=0.75,
                pad=0.03
            )

        if var_name is not None:
            _, _, _, cbar_label = self._get_variable_info(var_name)
            cbar.set_label(cbar_label)
        else:
            cbar.set_label(
                title.split('-')[-1].strip()
                if '-' in title
                else title
            )

        # Save with exact grid size
        plt.savefig(
            filename,
            dpi=150,
            bbox_inches='tight',
            pad_inches=0.1
        )

        plt.close()

    def plot_conditioning(
        self,
        cond_denorm,
        datetime_obj,
        var_names,
        lon_min=None,
        lon_max=None,
        lat_min=None,
        lat_max=None,
        output_subdir=None
    ):
        """
        Plot conditioning data with datetime
        
        Args:
            cond_denorm: Denormalized conditioning tensor
            datetime_obj: Datetime object for this conditioning data
            var_names: List of variable names
            lon_min, lon_max, lat_min, lat_max: Optional domain subset
            output_subdir: Optional subdirectory to save plots in
        """
        # Set default domain
        if lon_min is None:
            lon_min = float(self.lon.min())
        if lon_max is None:
            lon_max = float(self.lon.max())
        if lat_min is None:
            lat_min = float(self.lat.min())
        if lat_max is None:
            lat_max = float(self.lat.max())
        
        # Determine output directory
        if output_subdir:
            save_dir = output_subdir
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        # Format datetime for filename
        time_str = self._format_datetime(datetime_obj)
        safe_time = time_str.replace(" ", "_").replace(":", "").replace("-", "")
        
        # Get the conditioning sample (first one if multiple)
        cond_sample = cond_denorm[0] if cond_denorm.shape[0] > 0 else cond_denorm
        
        # Create plots for each conditioning variable
        for c, var_name in enumerate(var_names):
            if c >= cond_sample.shape[0]:
                break

            if var_name == "tp":
                plot_datetime = (
                    datetime_obj
                    - timedelta(
                        hours=24
                    )
                )
                plot_time_str = self._format_datetime(
                    plot_datetime
                )
                plot_safe_time = (
                    plot_time_str
                    .replace(" ", "_")
                    .replace(":", "")
                    .replace("-", "")
                )
            else:
                plot_time_str = time_str
                plot_safe_time = safe_time
            
            # Prepare data
            data_2d = cond_sample[c].cpu().detach().numpy()
            
            # Get plotting parameters
            levels, cmap, norm, label = self._get_variable_info(var_name)
            
            # Create filename with datetime
            filename = f"conditioning_{var_name}_{plot_safe_time}.png"
            save_path = os.path.join(save_dir, filename)
            
            # Create title with datetime
            title = f"Conditioning - {label}\nTime: {plot_time_str}"
            
            # Create plot
            self._create_geographic_plot(
                data_2d,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                cmap,
                norm,
                levels,
                title=title,
                filename=save_path,
                var_name=var_name
            )
            
            print(f"  Saved conditioning: {filename}")
            
    def plot_reference_targets(
        self,
        real_data_dict,
        datetime_obj,
        var_names,
        initialization_time=None,
        forecast_lead_hours=0,
        lon_min=None,
        lon_max=None,
        lat_min=None,
        lat_max=None,
        output_subdir=None
    ):
        """
        Plot reference target fields (CERRA) for the given datetime.

        real_data_dict: dict {var_name: np.ndarray(H,W)} (or torch tensor)
        var_names: order of variables to plot
        initialization_time: Forecast initialization datetime
        forecast_lead_hours: Forecast lead time in hours
        """
        # Set default domain
        if lon_min is None:
            lon_min = float(self.lon.min())
        if lon_max is None:
            lon_max = float(self.lon.max())
        if lat_min is None:
            lat_min = float(self.lat.min())
        if lat_max is None:
            lat_max = float(self.lat.max())

        # Determine output directory
        save_dir = output_subdir if output_subdir else self.output_dir
        os.makedirs(save_dir, exist_ok=True)

        # Format forecast-valid and initialization datetimes for filename/title
        time_str = self._format_datetime(datetime_obj)
        init_time_str = (
            self._format_datetime(initialization_time)
            if initialization_time is not None
            else "Unknown"
        )

        safe_time = time_str.replace(" ", "_").replace(":", "").replace("-", "")
        safe_init_time = (
            init_time_str.replace(" ", "_").replace(":", "").replace("-", "")
        )

        for var_name in var_names:
            if var_name not in real_data_dict:
                print(
                    f"  Warning: reference var "
                    f"'{var_name}' not in real_data_dict, skip."
                )
                continue

            data = real_data_dict[var_name]

            if isinstance(data, torch.Tensor):
                data_2d = data.cpu().detach().numpy()
            else:
                data_2d = np.asarray(data)

            # Get plotting parameters
            levels, cmap, norm, label = self._get_variable_info(var_name)

            filename = (
                f"reference_{var_name}_"
                f"init_{safe_init_time}_valid_{safe_time}_"
                f"lead_{int(forecast_lead_hours):03d}h.png"
            )
            save_path = os.path.join(save_dir, filename)

            title = (
                f"Reference - {label}\n"
                f"Initialization: {init_time_str}\n"
                f"Valid: {time_str}\n"
                f"Lead: +{int(forecast_lead_hours)} h"
            )

            self._create_geographic_plot(
                data_2d,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                cmap,
                norm,
                levels,
                title=title,
                filename=save_path,
                var_name=var_name
            )

            print(f"  Saved reference: {filename}")
