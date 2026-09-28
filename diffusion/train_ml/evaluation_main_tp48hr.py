#!/usr/bin/env python3
"""
evaluation
Date: 22 November 2025
"""

import argparse
import time
import os
import sys
import torch as th
import numpy as np
import pandas as pd
from datetime import datetime
import json
import glob
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# Import subroutines
from src_diffusion.evaluation_sample_tp48h import SampleGenerator
from src_diffusion.evaluation_statistics import StatisticsLogger
from src_diffusion.evaluation_plot_tp48h import PlotGenerator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate +24h forecast samples from multiple checkpoints and find best matches")
    
    # Model and data paths
    parser.add_argument("--checkpoint_dir", type=str, default=None, 
                        help="Directory containing checkpoint files (alternative to checkpoint_list)")
    parser.add_argument("--checkpoint_pattern", type=str, default="model*.pt", 
                        help="Pattern for checkpoint files (default: model*.pt)")
    parser.add_argument("--checkpoint_list", type=str, default=None,
                        help="File containing list of checkpoint paths (one per line)")
    parser.add_argument("--zarr_path", required=True, help="Path to Zarr store for coordinates and normalization")
    parser.add_argument(
        "--shapefile",
        type=str,
        default=None,
        help="India state/administrative boundary shapefile"
    )
    parser.add_argument("--output_dir", default="./checkpoint_comparison", help="Output directory")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate per checkpoint")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--in_channels", type=int, default=2, help="Number of input channels")
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Diffusion steps for sampling")
    parser.add_argument("--timestep_respacing", type=str, default="1000", help="Timestep respacing")
    
    #new_add
    parser.add_argument("--cond_channels", type=int, default=2, help="Number of conditioning channels")

    parser.add_argument("--diffusion_type", type=str, default="ddpm",
                        help="Diffusion type: ddpm or vp_sde")
    parser.add_argument("--sde_loss_type", type=str, default="score_mse",
                        help="SDE loss type")
    parser.add_argument("--beta_min", type=float, default=0.1,
                        help="Minimum beta for VP-SDE")
    parser.add_argument("--beta_max", type=float, default=20.0,
                        help="Maximum beta for VP-SDE")
    parser.add_argument("--sampling_eps", type=float, default=1e-3,
                        help="Minimum sampling time for SDE")
    parser.add_argument("--cfg_guidance_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale for sampling")

    # Model architecture (should match training)
    parser.add_argument("--num_channels", type=int, default=64, help="Number of model channels")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--attention_resolutions", type=str, default="16,8", help="Attention resolutions")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--learn_sigma", action="store_true", help="Learn sigma")
    parser.add_argument("--class_cond", action="store_true", help="Class conditioning")
    parser.add_argument("--noise_schedule", type=str, default="linear", help="Noise schedule")
    parser.add_argument("--use_scale_shift_norm", action="store_true", help="Use scale shift norm")
    
    # Conditioning options
    parser.add_argument("--use_real_cond", action="store_true", help="Use real conditioning from Zarr")
    parser.add_argument("--cond_datetime", type=str, default=None, 
                        help="Datetime for conditioning (e.g., '2015-01-15 12:00:00')")
    
    #new_add: seasonal
    parser.add_argument(
        "--add_seasonal_cond",
        action="store_true",
        help="Add seasonal encoding to conditioning",
    )
    parser.add_argument(
        "--seasonal_channels",
        type=int,
        default=0,
        help="Number of seasonal encoding channels",
    )

    parser.add_argument(
        "--forecast_lead_hours",
        type=int,
        default=0,
        help="Forecast lead time in hours for validation target (e.g. 24)",
    )
    
    # Variable names
    parser.add_argument("--input_vars", nargs="+", default=['t2m_era5_mean', 't2m_era5_std'],
                        help="Input variable names")
    parser.add_argument("--target_vars", nargs="+", default=['t2m_cerra_mean', 't2m_cerra_std'],
                        help="Target variable names")
    
    # Geographic domain (optional - defaults to full domain)
    parser.add_argument("--lon_min", type=float, default=None, help="Minimum longitude")
    parser.add_argument("--lon_max", type=float, default=None, help="Maximum longitude")
    parser.add_argument("--lat_min", type=float, default=None, help="Minimum latitude")
    parser.add_argument("--lat_max", type=float, default=None, help="Maximum latitude")
    
    # Validation options
    parser.add_argument("--save_best_samples", action="store_true", 
                        help="Save best matching samples separately")
    parser.add_argument("--summary_file", type=str, default="checkpoint_summary.txt", 
                        help="File to save checkpoint comparison summary")
    
    # Device
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    return parser.parse_args()


class DataLoader:
    """Helper class to load real data from Zarr"""

    def __init__(self, zarr_path):
        import zarr
        self.zarr_path = zarr_path
        self.store = zarr.open(zarr_path, mode='r')

    def find_index_by_datetime(self, target_datetime):
        """Find index for a specific datetime"""
        import pandas as pd
        import numpy as np

        if 'valid_time' not in self.store:
            raise ValueError("No 'valid_time' variable found in Zarr store")

        time_data = self.store['valid_time'][:]
        units = self.store['valid_time'].attrs.get('units', 'seconds since 1970-01-01')

        # Convert to datetime
        if 'seconds since' in units:
            datetimes = pd.to_datetime(time_data, unit='s', origin='unix', utc=True)
        else:
            datetimes = pd.to_datetime(time_data, unit='h', origin='unix', utc=True)

        target_ts = pd.Timestamp(target_datetime)
        if target_ts.tzinfo is None:
            target_ts = target_ts.tz_localize('UTC')
        else:
            target_ts = target_ts.tz_convert('UTC')

        time_diffs = np.abs((datetimes - target_ts).total_seconds())
        closest_idx = int(np.argmin(time_diffs))
        closest_datetime = datetimes[closest_idx]

        time_diff_seconds = float(time_diffs[closest_idx])
        if time_diff_seconds > 3600:
            raise ValueError(
                f"Requested datetime {target_ts} is not available in Zarr. "
                f"Closest datetime is {closest_datetime}, "
                f"which is {time_diff_seconds / 3600:.1f} hours away."
            )
        elif time_diff_seconds > 0:
            print(
                f" Warning: Closest datetime is "
                f"{time_diff_seconds / 3600:.2f} hours from target"
            )

        return closest_idx, closest_datetime

    def load_real_targets(self, datetime_str, target_vars):
        """Load real data for the given datetime"""
        idx, actual_dt = self.find_index_by_datetime(datetime_str)

        real_data = {}
        for var in target_vars:
            if var in self.store:
                data = self.store[var][idx].astype(np.float32)
                real_data[var] = data

                finite_points = int(np.isfinite(data).sum())
                total_points = int(data.size)

                print(f"  Loaded {var} shape: {data.shape}")
                print(f"    Valid points: {finite_points}/{total_points}")
                print(f"    NaN points: {int(np.isnan(data).sum())}")
                print(f"    Inf points: {int(np.isinf(data).sum())}")
            else:
                print(f"  Warning: {var} not found in Zarr")

        return real_data, actual_dt, idx

class ValidationMetrics:
    """Calculate and store validation metrics for a single checkpoint"""

    def __init__(self, output_dir, target_vars):
        self.output_dir = output_dir
        self.target_vars = target_vars
        self.metrics = {
            'conditioning_datetime': None,
            'forecast_valid_datetime': None,
            'actual_datetime': None,
            'zarr_index': None,
            'forecast_lead_hours': 0,
            'num_samples': 0,
            'samples': []
        }

    def set_metadata(
        self,
        cond_dt,
        forecast_valid_dt,
        actual_dt,
        zarr_idx,
        forecast_lead_hours
    ):
        """Set metadata for this validation run"""
        self.metrics['conditioning_datetime'] = str(cond_dt)
        self.metrics['forecast_valid_datetime'] = str(forecast_valid_dt)
        self.metrics['actual_datetime'] = str(actual_dt)
        self.metrics['zarr_index'] = int(zarr_idx)
        self.metrics['forecast_lead_hours'] = int(forecast_lead_hours)

    def calculate_sample_metrics(self, sample_idx, sample_tensor, real_data_dict):
        """
        Calculate pixel-wise differences and RMSE for a sample

        Args:
            sample_idx: Index of the sample
            sample_tensor: Tensor of shape [C, H, W]
            real_data_dict: Dictionary of real data arrays
        """
        sample_metrics = {
            'sample_idx': int(sample_idx),
            'per_variable': {},
            'overall_rmse': float('nan')
        }

        rmse_sum = 0.0
        num_vars = 0

        # Store all generated target variables for this sample.
        # For two target variables, the saved array will have shape [2, H, W].
        sample_variables_to_save = []

        for v, var_name in enumerate(self.target_vars):
            if var_name in real_data_dict:
                # Get real data
                real_np = np.asarray(real_data_dict[var_name])

                # Get sample data for this variable
                if v < sample_tensor.shape[0]:
                    sample_np = sample_tensor[v].cpu().detach().numpy()

                    if sample_np.shape != real_np.shape:
                        raise ValueError(
                            f"Shape mismatch for {var_name}: "
                            f"sample={sample_np.shape}, reference={real_np.shape}"
                        )

                    # Use only valid geographical grid points
                    valid_mask = np.isfinite(real_np) & np.isfinite(sample_np)
                    num_valid = int(np.sum(valid_mask))
                    total_points = int(real_np.size)

                    if num_valid == 0:
                        print(
                            f"Warning: No valid grid points for "
                            f"sample {sample_idx}, variable {var_name}"
                        )
                        continue

                    # Calculate difference only over valid points
                    diff_valid = sample_np[valid_mask] - real_np[valid_mask]
                    abs_diff = np.abs(diff_valid)

                    # Calculate statistics
                    rmse = np.sqrt(np.mean(diff_valid**2))
                    mae = np.mean(abs_diff)
                    max_diff = np.max(abs_diff)
                    min_diff = np.min(diff_valid)
                    max_positive = np.max(diff_valid)
                    max_negative = np.min(diff_valid)
                    mean_error = np.mean(diff_valid)
                    std_error = np.std(diff_valid)

                    # Store metrics
                    sample_metrics['per_variable'][var_name] = {
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'max_absolute_error': float(max_diff),
                        'min_error': float(min_diff),
                        'max_overestimate': float(max_positive),
                        'max_underestimate': float(max_negative),
                        'mean_error': float(mean_error),
                        'std_error': float(std_error),
                        'num_valid_points': num_valid,
                        'total_points': total_points,
                        'valid_fraction': float(num_valid / total_points)
                    }

                    rmse_sum += rmse
                    num_vars += 1

                    # Save the generated variable itself, not the difference.
                    # Invalid geographical points are kept as NaN.
                    sample_variable = np.full_like(
                        real_np,
                        np.nan,
                        dtype=np.float32
                    )
                    sample_variable[valid_mask] = sample_np[valid_mask].astype(
                        np.float32
                    )
                    sample_variables_to_save.append(sample_variable)

        # Save all target variables for this sample in ONE .npy file.
        # With two target variables the array shape is [2, H, W], where:
        #   data[0] = self.target_vars[0]
        #   data[1] = self.target_vars[1]
        if sample_variables_to_save:
            sample_variables_array = np.stack(
                sample_variables_to_save,
                axis=0
            ).astype(np.float32)

            sample_map_path = os.path.join(
                self.output_dir,
                f"sample_{sample_idx:03d}_all_variables.npy"
            )
            np.save(sample_map_path, sample_variables_array)
            sample_metrics['sample_map_path'] = sample_map_path

        # Calculate overall RMSE (average across variables)
        if num_vars > 0:
            sample_metrics['overall_rmse'] = float(rmse_sum / num_vars)

        self.metrics['samples'].append(sample_metrics)
        return sample_metrics

    def find_best_samples(self):
        """Find the best samples for each variable and overall"""
        if not self.metrics['samples']:
            return None

        best_samples = {
            'overall': {'sample_idx': None, 'rmse': float('inf')},
            'per_variable': {}
        }

        # Initialize per-variable best
        for var in self.target_vars:
            best_samples['per_variable'][var] = {
                'sample_idx': None,
                'rmse': float('inf')
            }

        # Find best samples
        for sample in self.metrics['samples']:
            # Check overall
            overall_rmse = sample['overall_rmse']
            if (
                np.isfinite(overall_rmse)
                and overall_rmse < best_samples['overall']['rmse']
            ):
                best_samples['overall']['sample_idx'] = sample['sample_idx']
                best_samples['overall']['rmse'] = overall_rmse

            # Check per variable
            for var, metrics in sample['per_variable'].items():
                rmse = metrics['rmse']
                if (
                    np.isfinite(rmse)
                    and rmse < best_samples['per_variable'][var]['rmse']
                ):
                    best_samples['per_variable'][var]['sample_idx'] = sample['sample_idx']
                    best_samples['per_variable'][var]['rmse'] = rmse

        self.metrics['best_samples'] = best_samples
        return best_samples

    def save_metrics(self, filename):
        """Save all metrics to text file"""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SAMPLE VALIDATION METRICS AGAINST REAL DATA\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Conditioning Datetime: {self.metrics['conditioning_datetime']}\n")
            f.write(f"Forecast Valid Datetime: {self.metrics['forecast_valid_datetime']}\n")
            f.write(f"Forecast Lead Hours: {self.metrics['forecast_lead_hours']}\n")
            f.write(f"Actual Datetime: {self.metrics['actual_datetime']}\n")
            f.write(f"Zarr Index: {self.metrics['zarr_index']}\n")
            f.write(f"Number of Samples: {self.metrics['num_samples']}\n\n")

            # Write per-sample metrics
            f.write("-" * 80 + "\n")
            f.write("PER-SAMPLE METRICS\n")
            f.write("-" * 80 + "\n\n")

            for sample in self.metrics['samples']:
                f.write(f"Sample {sample['sample_idx']:03d}:\n")

                if np.isfinite(sample['overall_rmse']):
                    f.write(f"  Overall RMSE: {sample['overall_rmse']:.6f}\n")
                else:
                    f.write("  Overall RMSE: N/A\n")

                for var, metrics in sample['per_variable'].items():
                    f.write(f"  {var}:\n")
                    f.write(f"    RMSE: {metrics['rmse']:.6f}\n")
                    f.write(f"    MAE: {metrics['mae']:.6f}\n")
                    f.write(f"    Max Error: {metrics['max_absolute_error']:.6f}\n")
                    f.write(f"    Mean Error: {metrics['mean_error']:.6f}\n")
                    f.write(f"    Std Error: {metrics['std_error']:.6f}\n")
                    f.write(
                        f"    Valid Points: "
                        f"{metrics['num_valid_points']}/{metrics['total_points']}\n"
                    )
                    f.write(f"    Valid Fraction: {metrics['valid_fraction']:.6f}\n")
                f.write("\n")

            # Write best samples
            if 'best_samples' in self.metrics:
                f.write("-" * 80 + "\n")
                f.write("BEST SAMPLES\n")
                f.write("-" * 80 + "\n\n")

                best = self.metrics['best_samples']

                if best['overall']['sample_idx'] is not None:
                    f.write(
                        f"Best Overall Sample: "
                        f"{best['overall']['sample_idx']:03d} "
                    )
                    f.write(f"(RMSE: {best['overall']['rmse']:.6f})\n\n")
                else:
                    f.write(
                        "Best Overall Sample: N/A "
                        "(no valid RMSE)\n\n"
                    )

                for var, info in best['per_variable'].items():
                    if info['sample_idx'] is not None:
                        f.write(
                            f"Best Sample for {var}: "
                            f"{info['sample_idx']:03d} "
                        )
                        f.write(f"(RMSE: {info['rmse']:.6f})\n")
                    else:
                        f.write(
                            f"Best Sample for {var}: "
                            f"N/A (no valid RMSE)\n"
                        )

        print(f"  Validation metrics saved to: {filepath}")

        # Also save as JSON for programmatic access
        json_path = filepath.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(
                self.metrics,
                f,
                indent=2,
                default=lambda x: (
                    float(x)
                    if isinstance(x, (np.floating, np.integer))
                    else x
                )
            )

        return filepath

class CheckpointValidator:
    """Validate samples from multiple checkpoints"""

    def __init__(self, output_dir, target_vars, lon_min, lon_max, lat_min, lat_max):
        self.output_dir = output_dir
        self.target_vars = target_vars
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.results = []
        self.best_overall = {
            'checkpoint': None,
            'sample_idx': None,
            'rmse': float('inf'),
            'sample': None,
            'per_variable': {}
        }

        # Initialize per-variable best
        for var in target_vars:
            self.best_overall['per_variable'][var] = {
                'checkpoint': None,
                'sample_idx': None,
                'rmse': float('inf'),
                'sample': None
            }

    def add_checkpoint_results(self, checkpoint_name, samples_denorm, real_data, metrics_file, plotter):
        """Add results from a checkpoint"""

        # Load metrics from file
        with open(metrics_file, 'r') as f:
            content = f.read()

        # Parse best samples from metrics
        best_overall_idx = None
        best_overall_rmse = float('inf')
        best_per_var = {}

        # Extract best overall sample
        overall_match = re.search(
            r"Best Overall Sample: (\d+) \(RMSE: ([\d.eE+-]+)\)",
            content
        )
        if overall_match:
            best_overall_idx = int(overall_match.group(1))
            best_overall_rmse = float(overall_match.group(2))

        # Extract best per variable
        for var in self.target_vars:
            var_match = re.search(
                rf"Best Sample for {re.escape(var)}: "
                rf"(\d+) \(RMSE: ([\d.eE+-]+)\)",
                content
            )
            if var_match:
                best_per_var[var] = {
                    'idx': int(var_match.group(1)),
                    'rmse': float(var_match.group(2))
                }

        # Store results
        result = {
            'checkpoint': checkpoint_name,
            'best_overall': {
                'idx': best_overall_idx,
                'rmse': best_overall_rmse,
                'sample': (
                    samples_denorm[best_overall_idx]
                    if (
                        best_overall_idx is not None
                        and best_overall_idx < len(samples_denorm)
                    )
                    else None
                )
            },
            'best_per_var': {}
        }

        # Update best overall across all checkpoints
        if (
            best_overall_idx is not None
            and np.isfinite(best_overall_rmse)
            and best_overall_rmse < self.best_overall['rmse']
        ):
            self.best_overall['checkpoint'] = checkpoint_name
            self.best_overall['sample_idx'] = best_overall_idx
            self.best_overall['rmse'] = best_overall_rmse
            self.best_overall['sample'] = result['best_overall']['sample']

        # Store per-variable best and update global best
        for var, info in best_per_var.items():
            result['best_per_var'][var] = info

            if (
                np.isfinite(info['rmse'])
                and info['rmse'] < self.best_overall['per_variable'][var]['rmse']
            ):
                self.best_overall['per_variable'][var]['checkpoint'] = checkpoint_name
                self.best_overall['per_variable'][var]['sample_idx'] = info['idx']
                self.best_overall['per_variable'][var]['rmse'] = info['rmse']

                if info['idx'] is not None and info['idx'] < len(samples_denorm):
                    self.best_overall['per_variable'][var]['sample'] = samples_denorm[info['idx']]

        self.results.append(result)
        return result

    def save_summary(self, filename):
        """Save checkpoint comparison summary"""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CHECKPOINT COMPARISON SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Write per-checkpoint results
            f.write("-" * 80 + "\n")
            f.write("PER-CHECKPOINT BEST SAMPLES\n")
            f.write("-" * 80 + "\n\n")

            for result in self.results:
                f.write(f"Checkpoint: {result['checkpoint']}\n")

                if result['best_overall']['idx'] is not None:
                    f.write(
                        f"  Best Overall - Sample "
                        f"{result['best_overall']['idx']:03d}, "
                    )
                    f.write(
                        f"RMSE: "
                        f"{result['best_overall']['rmse']:.6f}\n"
                    )
                else:
                    f.write("  Best Overall - None\n")

                for var, info in result['best_per_var'].items():
                    f.write(
                        f"  Best {var} - Sample {info['idx']:03d}, "
                        f"RMSE: {info['rmse']:.6f}\n"
                    )
                f.write("\n")

            # Write overall best across all checkpoints
            f.write("-" * 80 + "\n")
            f.write("BEST ACROSS ALL CHECKPOINTS\n")
            f.write("-" * 80 + "\n\n")

            f.write("Best Overall Sample:\n")
            f.write(f"  Checkpoint: {self.best_overall['checkpoint']}\n")

            if self.best_overall['sample_idx'] is not None:
                f.write(
                    f"  Sample Index: "
                    f"{self.best_overall['sample_idx']:03d}\n"
                )

            if np.isfinite(self.best_overall['rmse']):
                f.write(f"  RMSE: {self.best_overall['rmse']:.6f}\n\n")
            else:
                f.write("  RMSE: N/A\n\n")

            for var, info in self.best_overall['per_variable'].items():
                if info['checkpoint']:
                    f.write(f"Best {var}:\n")
                    f.write(f"  Checkpoint: {info['checkpoint']}\n")

                    if info['sample_idx'] is not None:
                        f.write(f"  Sample Index: {info['sample_idx']:03d}\n")

                    f.write(f"  RMSE: {info['rmse']:.6f}\n\n")

        print(f"Checkpoint comparison saved to: {filepath}")
        return filepath

    def save_best_overall_samples(self, plotter):
        """Save the best overall samples from all checkpoints"""
        best_dir = os.path.join(self.output_dir, 'best_overall')
        os.makedirs(best_dir, exist_ok=True)

        # Save overall best sample
        if self.best_overall['sample'] is not None:
            best_sample = self.best_overall['sample']
            checkpoint = self.best_overall['checkpoint']
            sample_idx = self.best_overall['sample_idx']

            # Save numpy array
            np.save(
                os.path.join(
                    best_dir,
                    f'best_overall_{checkpoint.replace(".pt", "")}_'
                    f'sample_{sample_idx:03d}.npy'
                ),
                best_sample.cpu().numpy()
            )

            # Create plot for best overall
            safe_checkpoint = checkpoint.replace('.pt', '').replace('model', 'ckpt')

            for c, var in enumerate(self.target_vars):
                if c < best_sample.shape[0]:
                    data_2d = best_sample[c].cpu().numpy()

                    # Use the same plotting scale as reference/generated plots
                    levels, cmap, norm, label = plotter._get_variable_info(var)

                    plotter._create_geographic_plot(
                        data_2d,
                        self.lon_min,
                        self.lon_max,
                        self.lat_min,
                        self.lat_max,
                        cmap=cmap,
                        norm=norm,
                        levels=levels,
                        title=(
                            f"Best Overall - {label}\n"
                            f"Checkpoint: {checkpoint}"
                        ),
                        filename=os.path.join(
                            best_dir,
                            f'best_overall_{var}_{safe_checkpoint}.png'
                        )
                    )

            print(f"  Saved best overall sample from {checkpoint}")

        # Save per-variable best samples
        for var, info in self.best_overall['per_variable'].items():
            if info.get('sample') is not None:
                best_sample = info['sample']
                checkpoint = info['checkpoint']
                sample_idx = info['sample_idx']

                # Find variable index
                var_idx = self.target_vars.index(var) if var in self.target_vars else -1

                if var_idx >= 0 and var_idx < best_sample.shape[0]:
                    var_data = best_sample[var_idx].cpu().numpy()

                    np.save(
                        os.path.join(
                            best_dir,
                            f'best_{var}_{checkpoint.replace(".pt", "")}_'
                            f'sample_{sample_idx:03d}.npy'
                        ),
                        var_data
                    )

                    # Use the same plotting scale as reference/generated plots
                    levels, cmap, norm, label = plotter._get_variable_info(var)

                    # Create plot
                    safe_checkpoint = checkpoint.replace('.pt', '').replace('model', 'ckpt')

                    plotter._create_geographic_plot(
                        var_data,
                        self.lon_min,
                        self.lon_max,
                        self.lat_min,
                        self.lat_max,
                        cmap=cmap,
                        norm=norm,
                        levels=levels,
                        title=(
                            f"Best {label}\n"
                            f"Checkpoint: {checkpoint}"
                        ),
                        filename=os.path.join(
                            best_dir,
                            f'best_{var}_{safe_checkpoint}.png'
                        )
                    )

        # Create comparison table
        self._create_comparison_table(best_dir)

    def _create_comparison_table(self, best_dir):
        """Create a text table comparing all checkpoints"""
        table_path = os.path.join(best_dir, 'checkpoint_comparison.txt')

        with open(table_path, 'w') as f:
            f.write("CHECKPOINT COMPARISON TABLE\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Checkpoint':<15} {'Best Sample':<12} {'Overall RMSE':<15} ")

            for var in self.target_vars:
                f.write(f"{var[:20]:<22} ")

            f.write("\n")
            f.write("-" * 100 + "\n")

            for result in self.results:
                ckpt = result['checkpoint'].replace('model', '').replace('.pt', '')

                if len(ckpt) > 10:
                    ckpt = ckpt[:10]

                f.write(f"{ckpt:<15} ")

                if result['best_overall']['idx'] is not None:
                    f.write(f"{result['best_overall']['idx']:03d}{'':9} ")
                    f.write(f"{result['best_overall']['rmse']:<15.6f} ")
                else:
                    f.write(f"{'N/A':<12} {'N/A':<15} ")

                for var in self.target_vars:
                    if var in result['best_per_var']:
                        f.write(f"{result['best_per_var'][var]['idx']:03d} ")
                        f.write(
                            f"({result['best_per_var'][var]['rmse']:.4f})"
                            f"{'':6} "
                        )
                    else:
                        f.write(f"{'':22} ")

                f.write("\n")

            f.write("-" * 100 + "\n")
            f.write("\nBEST OVERALL:\n")
            f.write(f"  Checkpoint: {self.best_overall['checkpoint']}\n")

            if self.best_overall['sample_idx'] is not None:
                f.write(f"  Sample: {self.best_overall['sample_idx']:03d}\n")

            if np.isfinite(self.best_overall['rmse']):
                f.write(f"  RMSE: {self.best_overall['rmse']:.6f}\n")
            else:
                f.write("  RMSE: N/A\n")

        print(f"  Comparison table saved to: {table_path}")

def get_checkpoint_files(checkpoint_dir, pattern):
    """Get sorted list of checkpoint files from directory"""
    search_pattern = os.path.join(checkpoint_dir, pattern)
    files = glob.glob(search_pattern)
    
    # Extract step numbers and sort
    def get_step_number(filename):
        match = re.search(r'model(\d+)\.pt', os.path.basename(filename))
        return int(match.group(1)) if match else 0
    
    files.sort(key=get_step_number)
    return files


def load_checkpoint_list(checkpoint_list_file):
    """Load checkpoint paths from a list file"""
    with open(checkpoint_list_file, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    # Verify files exist
    existing_files = []
    for f in files:
        if os.path.exists(f):
            existing_files.append(f)
        else:
            print(f"Warning: Checkpoint file not found: {f}")
    
    return existing_files


def main():
    """Main execution function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of checkpoint files
    if args.checkpoint_list:
        checkpoint_files = load_checkpoint_list(args.checkpoint_list)
        print(f"\n{'='*80}")
        print(f"Loaded {len(checkpoint_files)} checkpoint files from list: {args.checkpoint_list}")
    elif args.checkpoint_dir:
        checkpoint_files = get_checkpoint_files(args.checkpoint_dir, args.checkpoint_pattern)
        print(f"\n{'='*80}")
        print(f"Found {len(checkpoint_files)} checkpoint files in {args.checkpoint_dir}")
    else:
        raise ValueError("Must provide either --checkpoint_dir or --checkpoint_list")
    
    # Display checkpoint files
    for i, f in enumerate(checkpoint_files):
        print(f"  {i+1:2d}. {os.path.basename(f)}")
    print(f"{'='*80}")
    
    if not checkpoint_files:
        raise ValueError("No checkpoint files found")
    
    # Setup device
    device = th.device(args.device) if args.device else th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Initialize plot generator (loads coordinates once)
    plotter = PlotGenerator(
        args.zarr_path,
        args.output_dir,
        shapefile=args.shapefile
    )
    
    # Initialize data loader for validation
    data_loader = DataLoader(args.zarr_path)
    
    # Load real target data for validation at forecast valid time
    print(f"\n{'='*80}")
    print("LOADING REAL TARGET DATA FOR FORECAST VALIDATION")
    print(f"{'='*80}")

    if args.use_real_cond and args.cond_datetime:
        initialization_datetime = pd.Timestamp(args.cond_datetime)

        if initialization_datetime.tzinfo is None:
            initialization_datetime = initialization_datetime.tz_localize('UTC')
        else:
            initialization_datetime = initialization_datetime.tz_convert('UTC')

        forecast_valid_datetime = (
            initialization_datetime
            + pd.Timedelta(hours=args.forecast_lead_hours)
        )

        print(f"Initialization datetime: {initialization_datetime}")
        print(f"Forecast lead hours: {args.forecast_lead_hours}")
        print(f"Forecast valid datetime: {forecast_valid_datetime}")

        real_data, actual_datetime, zarr_index = data_loader.load_real_targets(
            forecast_valid_datetime,
            args.target_vars
        )

        print(
            f"Found real target data at index {zarr_index} "
            f"(datetime: {actual_datetime})"
        )

        # Build a common valid geographical mask from ALL target variables.
        # A point is valid only where every target variable is finite.
        valid_mask = None

        for mask_var in args.target_vars:

            if mask_var not in data_loader.store:
                raise ValueError(
                    f"Target mask variable '{mask_var}' not found in Zarr store"
                )

            mask_data = data_loader.store[
                mask_var
            ][zarr_index].astype(np.float32)

            current_mask = np.isfinite(
                mask_data
            )

            print(
                f"Valid mask for {mask_var}: "
                f"{int(current_mask.sum())}/{current_mask.size} valid grid points"
            )

            if valid_mask is None:
                valid_mask = current_mask
            else:
                valid_mask = (
                    valid_mask
                    & current_mask
                )

        if valid_mask is None:
            raise ValueError(
                "Could not construct valid geographical mask from target variables"
            )

        print(
            f"Combined valid plotting/evaluation mask from "
            f"{args.target_vars}: "
            f"{int(valid_mask.sum())}/{valid_mask.size} valid grid points"
        )

        plotter.set_valid_mask(
            valid_mask
        )
    else:
        raise ValueError("Must provide --cond_datetime with --use_real_cond for validation")

    # Initialize checkpoint validator
    validator = CheckpointValidator(
        args.output_dir, 
        args.target_vars,
        args.lon_min, args.lon_max, args.lat_min, args.lat_max
    )
    
    #new_add: multiple variables
    auto_in_channels = len(args.target_vars)
    auto_cond_channels = len(args.input_vars) + (
        args.seasonal_channels if args.add_seasonal_cond else 0
    )

    print(f"Auto evaluation in_channels = {auto_in_channels}")
    print(f"Auto evaluation cond_channels = {auto_cond_channels}")
    print(f"Physical cond vars = {args.input_vars}")
    print(f"Target vars = {args.target_vars}")
    print(f"Seasonal channels = {args.seasonal_channels if args.add_seasonal_cond else 0}")
    print(f"Forecast lead hours = {args.forecast_lead_hours}")

    # Build base model configuration (same for all checkpoints)
    base_model_config = {
        'image_size': args.image_size,
        'class_cond': args.class_cond,
        'learn_sigma': args.learn_sigma,
        'sigma_small': False,
        'num_channels': args.num_channels,
        'num_res_blocks': args.num_res_blocks,
        'num_heads': args.num_heads,
        'num_heads_upsample': -1,
        'attention_resolutions': args.attention_resolutions,
        'dropout': args.dropout,
        'diffusion_steps': args.diffusion_steps,
        'noise_schedule': args.noise_schedule,
        'timestep_respacing': args.timestep_respacing,
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': True,
        'rescale_learned_sigmas': True,
        'use_checkpoint': False,
        'use_scale_shift_norm': args.use_scale_shift_norm,
        'in_channels': auto_in_channels,
        'cond_channels': auto_cond_channels,
        'diffusion_type': args.diffusion_type,
        'sde_loss_type': args.sde_loss_type,
        'beta_min': args.beta_min,
        'beta_max': args.beta_max,
        'sampling_eps': args.sampling_eps,
        'cfg_guidance_scale': args.cfg_guidance_scale,#new_add
    }
    
    # Process each checkpoint
    for ckpt_idx, checkpoint_path in enumerate(checkpoint_files):
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\n{'='*80}")
        print(f"PROCESSING CHECKPOINT {ckpt_idx+1}/{len(checkpoint_files)}: {checkpoint_name}")
        print(f"{'='*80}")
        
        # Create checkpoint-specific output subdirectory
        ckpt_output = os.path.join(args.output_dir, f"checkpoint_{checkpoint_name.replace('.pt', '')}")
        os.makedirs(ckpt_output, exist_ok=True)
        
        # Initialize sample generator for this checkpoint
        generator = SampleGenerator(device)
        
        # Load model
        generator.load_model(checkpoint_path, base_model_config)
        
        # Load normalizers
        generator.load_normalizers(args.zarr_path, args.input_vars, args.target_vars)
        
        # Create conditioning data
        if args.use_real_cond:
            cond_norm, cond_denorm, conditioning_datetime = generator.load_real_conditioning_by_datetime(
                args.zarr_path, args.input_vars, args.cond_datetime, args.num_samples,
                add_seasonal_cond=args.add_seasonal_cond, #new_add
                seasonal_channels=args.seasonal_channels,
                forecast_valid_datetime=forecast_valid_datetime,
            )
        else:
            cond_norm, cond_denorm = generator.create_random_conditioning(
                args.num_samples, len(args.input_vars), args.image_size, args.image_size,
                add_seasonal_cond=args.add_seasonal_cond, #new_add
                seasonal_channels=args.seasonal_channels,
            )
            conditioning_datetime = datetime.now()
        
        # Generate samples
        samples_norm, sampling_time = generator.generate_samples(
            cond_norm, args.num_samples, progress=True
        )
        
        # Denormalize samples
        samples_denorm = generator.denormalize_samples(samples_norm)

        valid_mask_tensor = th.from_numpy(
            valid_mask
        ).to(
            device=samples_denorm.device,
            dtype=th.bool
        )

        samples_denorm = samples_denorm.masked_fill(
            ~valid_mask_tensor.unsqueeze(0).unsqueeze(0),
            float("nan")
        )

        print(
            f"Applied combined target valid-land mask "
            f"{args.target_vars} to generated samples. "
            f"Invalid/ocean points are NaN."
        )
        
        # Create validation metrics for this checkpoint
        validator_metrics = ValidationMetrics(ckpt_output, args.target_vars)
        validator_metrics.set_metadata(
            args.cond_datetime,
            forecast_valid_datetime,
            actual_datetime,
            zarr_index,
            args.forecast_lead_hours
        )
        validator_metrics.metrics['num_samples'] = args.num_samples
        
        # Calculate metrics for each sample
        for i in range(args.num_samples):
            validator_metrics.calculate_sample_metrics(i, samples_denorm[i], real_data)
        
        # Find best samples
        validator_metrics.find_best_samples()
        
        # Save metrics
        metrics_filename = f"validation_metrics_{checkpoint_name.replace('.pt', '')}.txt"
        metrics_file = validator_metrics.save_metrics(metrics_filename)
        
        # Add results to checkpoint validator
        validator.add_checkpoint_results(
            checkpoint_name, 
            samples_denorm, 
            real_data, 
            metrics_file,
            plotter
        )
        
        # Generate plots for this checkpoint
        plotter.set_conditioning_datetime(conditioning_datetime)
        
        # Plot each sample separately
        for i in range(min(args.num_samples, 5)):  # Plot first 5 samples
            plotter.plot_mean_and_std_separate(
                samples_denorm, i, args.target_vars,
                conditioning_time=conditioning_datetime,
                forecast_valid_time=forecast_valid_datetime,
                forecast_lead_hours=args.forecast_lead_hours,
                lon_min=args.lon_min, lon_max=args.lon_max, 
                lat_min=args.lat_min, lat_max=args.lat_max,
                output_subdir=ckpt_output
            )
        
        # Also create grid plot for this checkpoint
        plotter.plot_all_samples_grid(
            samples_denorm, args.target_vars,
            conditioning_time=conditioning_datetime,
            forecast_valid_time=forecast_valid_datetime,
            forecast_lead_hours=args.forecast_lead_hours,
            lon_min=args.lon_min, lon_max=args.lon_max, 
            lat_min=args.lat_min, lat_max=args.lat_max,
            output_subdir=ckpt_output
        )
        
        # Conditioning-data PNG plotting disabled.
        # plotter.plot_conditioning(
        #     cond_denorm, conditioning_datetime, args.input_vars,
        #     lon_min=args.lon_min, lon_max=args.lon_max,
        #     lat_min=args.lat_min, lat_max=args.lat_max,
        #     output_subdir=ckpt_output
        # )
        # Plot reference target data at the forecast valid datetime
        plotter.plot_reference_targets(
            real_data, actual_datetime, args.target_vars,
            initialization_time=conditioning_datetime,
            forecast_lead_hours=args.forecast_lead_hours,
            lon_min=args.lon_min, lon_max=args.lon_max,
            lat_min=args.lat_min, lat_max=args.lat_max,
            output_subdir=ckpt_output
        )  
        print(f"Completed checkpoint {checkpoint_name}")
    
    # Save overall comparison summary
    summary_file = validator.save_summary(args.summary_file)
    
    # Save best overall samples
    if args.save_best_samples:
        validator.save_best_overall_samples(plotter)
    
    print(f"\n{'='*80}")
    print(f"CHECKPOINT COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processed {len(checkpoint_files)} checkpoints")
    print(f"Summary file: {summary_file}")
    if validator.best_overall['checkpoint']:
        print(f"Best overall checkpoint: {validator.best_overall['checkpoint']}")
        if validator.best_overall['sample_idx'] is not None:
            print(f"Best overall sample index: {validator.best_overall['sample_idx']}")
        print(f"Best overall RMSE: {validator.best_overall['rmse']:.6f}")
    print(f"Best samples saved in: {args.output_dir}/best_overall/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
