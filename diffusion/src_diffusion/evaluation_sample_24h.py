#!/usr/bin/env python3
"""
Sample Generation Subroutine
"""

import torch as th
import torch.nn.functional as F
import time
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src_diffusion.diffusion_dist import create_model_and_diffusion
from src_diffusion.zarr_load import ZarrNormalizer, get_vars_indices
import zarr
import numpy as np


class SampleGenerator:
    """Handles model loading and sample generation"""
    
    def __init__(self, device=None):
        self.device = device or th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model = None
        self.diffusion = None
        self.target_normalizer = None
        self.input_normalizer = None

        #new_add: store model and original spatial sizes
        self.image_size = None
        self.original_height = None
        self.original_width = None

        print(f"SampleGenerator initialized with device: {self.device}")
    
    def load_model(self, checkpoint_path, model_config):
        """Load model from checkpoint"""
        print(f"\n{'='*60}")
        print(f"Loading model from: {checkpoint_path}")
        print(f"{'='*60}")
        
        #new_add: store image size used during training
        self.image_size = int(model_config.get("image_size", 256))
        print(f"Model image size: {self.image_size}")
        
        # Create model with the same configuration as training
        self.model, self.diffusion = create_model_and_diffusion(**model_config)
        
        # Load checkpoint
        checkpoint = th.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            print("Loaded model state from checkpoint")

        elif isinstance(checkpoint, dict) and 'ema' in checkpoint:
            self.model.load_state_dict(checkpoint['ema'])
            print("Loaded EMA model from checkpoint")

        #new_add
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model_state_dict from checkpoint")

        else:
            self.model.load_state_dict(checkpoint)
            print("Loaded checkpoint directly")
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model moved to device: {self.device}")
        print(f"Model in_channels: {self.model.in_channels}")

        if hasattr(self.model, "cond_channels"):
            print(f"Model cond_channels: {self.model.cond_channels}")
        
        # Get step information if available
        if isinstance(checkpoint, dict) and 'step' in checkpoint:
            print(f"Checkpoint step: {checkpoint['step']}")
        
        return self.model, self.diffusion
    
    def load_normalizers(self, zarr_path, input_vars, target_vars):
        """Load normalizers from Zarr metadata"""
        print(f"\n{'='*60}")
        print("Loading normalizers from Zarr metadata")
        print(f"{'='*60}")
        
        # Open Zarr store
        z = zarr.open(zarr_path, mode='r')
        
        # Get variable indices
        def get_vars_indices(vars_list, z):
            indices = []
            all_vars = np.array(z['variable'])

            for v in vars_list:
                found = False

                for j, av in enumerate(all_vars):
                    if v == av:
                        indices.append(j)
                        found = True
                        break

                if not found:
                    raise ValueError(
                        f"Variable '{v}' not found in Zarr variable metadata"
                    )

            return indices
        
        target_indices = get_vars_indices(target_vars, z)
        input_indices = get_vars_indices(input_vars, z)
        
        # Load statistics and move to device
        min_stats = th.from_numpy(
            z['min'][:].astype(np.float32)
        ).to(self.device)

        max_stats = th.from_numpy(
            z['max'][:].astype(np.float32)
        ).to(self.device)
        
        # Extract statistics
        target_min = min_stats[target_indices]
        target_max = max_stats[target_indices]

        input_min = min_stats[input_indices]
        input_max = max_stats[input_indices]
        
        # Create normalizers
        self.target_normalizer = ZarrNormalizer(
            min_val=target_min,
            max_val=target_max
        )

        self.input_normalizer = ZarrNormalizer(
            min_val=input_min,
            max_val=input_max
        )
        
        print(f"Input variables: {input_vars}")
        print(f"Input min shape: {input_min.shape}")
        print(f"Input max shape: {input_max.shape}")

        print(f"Target variables: {target_vars}")
        print(f"Target min shape: {target_min.shape}")
        print(f"Target max shape: {target_max.shape}")

        print("\nInput normalization ranges:")
        for i, var in enumerate(input_vars):
            print(
                f"  {var}: "
                f"min={input_min[i].item():.6g}, "
                f"max={input_max[i].item():.6g}"
            )

        print("\nTarget normalization ranges:")
        for i, var in enumerate(target_vars):
            print(
                f"  {var}: "
                f"min={target_min[i].item():.6g}, "
                f"max={target_max[i].item():.6g}"
            )
        
        return self.target_normalizer, self.input_normalizer
    
    def find_index_by_datetime(self, zarr_path, target_datetime):
        """
        Find the index in Zarr store corresponding to a specific datetime
        
        Args:
            zarr_path: Path to Zarr store
            target_datetime: Datetime string (e.g., "2015-01-15 12:00:00")
            
        Returns:
            index: The index in the Zarr array
            actual_datetime: The actual datetime at that index
        """
        store = zarr.open(zarr_path, mode='r')
        
        if 'valid_time' not in store:
            raise ValueError("No 'valid_time' variable found in Zarr store")
        
        # Load time data
        time_data = store['valid_time'][:]
        units = store['valid_time'].attrs.get(
            'units',
            'seconds since 1970-01-01'
        )
        
        # Convert to datetime
        if 'seconds since' in units:
            datetimes = pd.to_datetime(
                time_data,
                unit='s',
                origin='unix',
                utc=True
            )
        else:
            # Try other common units
            datetimes = pd.to_datetime(
                time_data,
                unit='h',
                origin='unix',
                utc=True
            )
        
        # Parse target datetime
        target_ts = pd.Timestamp(target_datetime)

        if target_ts.tzinfo is None:
            target_ts = target_ts.tz_localize('UTC')
        else:
            target_ts = target_ts.tz_convert('UTC')
        
        # Find closest index
        time_diffs = np.abs(
            (datetimes - target_ts).total_seconds()
        )

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
        
        print(f"Target datetime: {target_ts}")
        print(f"Found index {closest_idx} with datetime: {closest_datetime}")
        
        return closest_idx, closest_datetime
    
    def load_real_conditioning_by_datetime(
        self, 
        zarr_path, 
        input_vars, 
        datetime_str, 
        num_samples,
        add_seasonal_cond=False, #new_add
        seasonal_channels=0,
        forecast_valid_datetime=None,
    ):
        """
        Load real conditioning data from Zarr by datetime
        
        Args:
            zarr_path: Path to Zarr store
            input_vars: List of input variable names
            datetime_str: Initialization datetime string (e.g., "2015-01-15 12:00:00")
            num_samples: Number of samples to generate
            forecast_valid_datetime: Forecast valid datetime used for seasonal encoding
            
        Returns:
            cond_norm: Normalized conditioning tensor
            cond_denorm: Denormalized conditioning tensor
            actual_datetime: The actual datetime used
        """
        # Find index for datetime
        index, actual_datetime = self.find_index_by_datetime(
            zarr_path,
            datetime_str
        )
        
        print(
            f"\nLoading real conditioning data from index "
            f"{index} (datetime: {actual_datetime})"
        )

        if forecast_valid_datetime is not None:
            print(
                f"Forecast valid datetime for seasonal encoding: "
                f"{forecast_valid_datetime}"
            )

        store = zarr.open(zarr_path, mode='r')
        
        # Load conditioning data
        cond_data = []

        for var in input_vars:

            if var not in store:
                raise ValueError(
                    f"Conditioning variable '{var}' not found in Zarr store"
                )

            if var == "tp":

                if forecast_valid_datetime is None:
                    raise ValueError(
                        "forecast_valid_datetime is required "
                        "for 24-hour-lagged precipitation"
                    )

                precipitation_time = (
                    pd.Timestamp(
                        forecast_valid_datetime
                    )
                    - pd.Timedelta(
                        hours=24
                    )
                )

                if precipitation_time.tzinfo is None:
                    precipitation_time = (
                        precipitation_time
                        .tz_localize(
                            "UTC"
                        )
                    )
                else:
                    precipitation_time = (
                        precipitation_time
                        .tz_convert(
                            "UTC"
                        )
                    )

                precipitation_index, precipitation_datetime = (
                    self.find_index_by_datetime(
                        zarr_path,
                        precipitation_time
                    )
                )

                data = (
                    store[var][
                        precipitation_index
                    ]
                    .astype(
                        np.float32
                    )
                )

            else:

                data = store[var][index].astype(np.float32)

            print(
                f"  Loaded {var}: "
                f"shape={data.shape}, "
                f"NaN={np.isnan(data).sum()}, "
                f"Inf={np.isinf(data).sum()}"
            )

            cond_data.append(data)
        
        cond_denorm = th.from_numpy(
            np.stack(cond_data, axis=0)
        ).unsqueeze(0).float().to(self.device)

        #new_add: remember original geographical dimensions
        self.original_height = cond_denorm.shape[-2]
        self.original_width = cond_denorm.shape[-1]

        print(
            f"Original conditioning size: "
            f"{self.original_height} x {self.original_width}"
        )

        print(
            f"Original conditioning tensor shape: "
            f"{cond_denorm.shape}"
        )

        # Normalize exactly as training
        cond_norm = self.input_normalizer.normalize(
            cond_denorm.clone()
        )

        #new_add: same NaN/Inf handling used during training
        cond_norm = th.nan_to_num(
            cond_norm,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        print(
            f"Normalized physical conditioning shape: "
            f"{cond_norm.shape}"
        )

        print(
            f"Normalized conditioning finite: "
            f"{th.isfinite(cond_norm).all().item()}"
        )

        #new_add: seasonal conditioning
        if add_seasonal_cond and seasonal_channels > 0:

            if forecast_valid_datetime is None:
                seasonal_datetime = actual_datetime
            else:
                seasonal_datetime = pd.Timestamp(
                    forecast_valid_datetime
                )

                if seasonal_datetime.tzinfo is None:
                    seasonal_datetime = seasonal_datetime.tz_localize(
                        'UTC'
                    )
                else:
                    seasonal_datetime = seasonal_datetime.tz_convert(
                        'UTC'
                    )

            print(
                f"Seasonal conditioning datetime: "
                f"{seasonal_datetime}"
            )

            day_of_year = seasonal_datetime.dayofyear

            seasonal_features = []
            frequencies = [1, 2, 3, 4, 6, 12]

            for freq in frequencies:
                angle = (
                    2
                    * np.pi
                    * freq
                    * (day_of_year - 1)
                    / 365.25
                )

                seasonal_features.append(
                    np.sin(angle)
                )

                seasonal_features.append(
                    np.cos(angle)
                )

            seasonal_features = seasonal_features[
                :seasonal_channels
            ]

            if len(seasonal_features) != seasonal_channels:
                raise ValueError(
                    f"Requested {seasonal_channels} seasonal channels, "
                    f"but only {len(seasonal_features)} were generated"
                )

            h, w = cond_norm.shape[-2], cond_norm.shape[-1]

            seasonal_tensor = th.tensor(
                seasonal_features,
                dtype=th.float32,
                device=self.device
            ).view(
                1,
                -1,
                1,
                1
            ).expand(
                1,
                -1,
                h,
                w
            )

            cond_norm = th.cat(
                [
                    cond_norm,
                    seasonal_tensor
                ],
                dim=1
            )

            print(
                f"Conditioning shape after seasonal channels: "
                f"{cond_norm.shape}"
            )

        #new_add: verify channel number matches model
        if hasattr(self.model, "cond_channels"):

            if cond_norm.shape[1] != self.model.cond_channels:

                raise ValueError(
                    f"Condition channel mismatch: "
                    f"model expects {self.model.cond_channels}, "
                    f"but evaluation conditioning has "
                    f"{cond_norm.shape[1]} channels"
                )

        #new_add: pad conditioning to same image size used during training
        if self.image_size is None:

            raise ValueError(
                "Model image_size is not available. "
                "Call load_model() before loading conditioning data."
            )

        h = cond_norm.shape[-2]
        w = cond_norm.shape[-1]

        pad_h = self.image_size - h
        pad_w = self.image_size - w

        if pad_h < 0 or pad_w < 0:

            raise ValueError(
                f"Conditioning size {h}x{w} is larger than "
                f"model image_size {self.image_size}"
            )

        if pad_h > 0 or pad_w > 0:

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            print(
                f"Padding conditioning: "
                f"left={pad_left}, right={pad_right}, "
                f"top={pad_top}, bottom={pad_bottom}"
            )

            cond_norm = F.pad(
                cond_norm,
                (
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom
                ),
                mode="replicate"
            )

        print(
            f"Padded normalized conditioning shape: "
            f"{cond_norm.shape}"
        )

        if (
            cond_norm.shape[-2] != self.image_size
            or cond_norm.shape[-1] != self.image_size
        ):

            raise ValueError(
                f"Padding failed: conditioning shape is "
                f"{cond_norm.shape[-2]}x{cond_norm.shape[-1]}, "
                f"expected {self.image_size}x{self.image_size}"
            )

        if not th.isfinite(cond_norm).all():

            raise ValueError(
                "Non-finite values remain in normalized conditioning "
                "after NaN/Inf cleaning"
            )
        
        # Repeat to match num_samples if needed
        if cond_norm.shape[0] < num_samples:

            cond_norm = cond_norm.repeat(
                num_samples,
                1,
                1,
                1
            )

            cond_denorm = cond_denorm.repeat(
                num_samples,
                1,
                1,
                1
            )

        elif cond_norm.shape[0] > num_samples:

            cond_norm = cond_norm[
                :num_samples
            ]

            cond_denorm = cond_denorm[
                :num_samples
            ]

        print(
            f"Final conditioning shape for sampling: "
            f"{cond_norm.shape}"
        )

        print(
            f"Original-scale conditioning shape for plotting: "
            f"{cond_denorm.shape}"
        )
        
        return cond_norm, cond_denorm, actual_datetime
    
    def create_random_conditioning(
        self, num_samples, channels, height, width,
        add_seasonal_cond=False, seasonal_channels=0, #new_add
    ):
        """Create random conditioning data"""

        #new_add: remember original dimensions
        self.original_height = height
        self.original_width = width

        # Create random data in denormalized space
        cond_denorm = th.randn(
            num_samples,
            channels,
            height,
            width
        ).to(self.device)
        
        # Scale to realistic ranges
        if channels == 2:  # [mean, std]

            cond_denorm[:, 0, :, :] = (
                cond_denorm[:, 0, :, :]
                * 10
                + 285
            )

            cond_denorm[:, 1, :, :] = (
                th.abs(
                    cond_denorm[:, 1, :, :]
                )
                * 5
                + 2
            )
        
        # Normalize
        cond_norm = self.input_normalizer.normalize(
            cond_denorm.clone()
        )

        #new_add: same NaN/Inf handling used during training
        cond_norm = th.nan_to_num(
            cond_norm,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        #new_add
        if add_seasonal_cond and seasonal_channels > 0:

            seasonal_tensor = th.zeros(
                num_samples,
                seasonal_channels,
                height,
                width,
                device=self.device
            )

            cond_norm = th.cat(
                [
                    cond_norm,
                    seasonal_tensor
                ],
                dim=1
            )

        #new_add: verify conditioning channels
        if hasattr(self.model, "cond_channels"):

            if cond_norm.shape[1] != self.model.cond_channels:

                raise ValueError(
                    f"Condition channel mismatch: "
                    f"model expects {self.model.cond_channels}, "
                    f"but random conditioning has "
                    f"{cond_norm.shape[1]} channels"
                )

        #new_add: pad random conditioning if needed
        if self.image_size is not None:

            h = cond_norm.shape[-2]
            w = cond_norm.shape[-1]

            pad_h = self.image_size - h
            pad_w = self.image_size - w

            if pad_h < 0 or pad_w < 0:

                raise ValueError(
                    f"Random conditioning size {h}x{w} "
                    f"is larger than model image_size "
                    f"{self.image_size}"
                )

            if pad_h > 0 or pad_w > 0:

                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top

                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                cond_norm = F.pad(
                    cond_norm,
                    (
                        pad_left,
                        pad_right,
                        pad_top,
                        pad_bottom
                    ),
                    mode="replicate"
                )
        
        return cond_norm, cond_denorm
    
    @th.no_grad()
    def generate_samples(self, cond_tensor, num_samples, progress=True):
        """Generate samples using the diffusion model"""
        print(f"\n{'='*60}")
        print("Generating samples...")
        print(f"{'='*60}")
        
        self.model.eval()
        
        # Ensure cond_tensor has the right batch size
        if cond_tensor.shape[0] < num_samples:

            cond_tensor = cond_tensor.repeat(
                num_samples,
                1,
                1,
                1
            )

        elif cond_tensor.shape[0] > num_samples:

            cond_tensor = cond_tensor[
                :num_samples
            ]
        
        cond_tensor = cond_tensor.to(self.device)

        #cond_tensor = th.zeros_like(cond_tensor) #test for zero_cond
        
        print(
            f"Conditioning shape: "
            f"{cond_tensor.shape}"
        )

        print(
            f"Conditioning finite: "
            f"{th.isfinite(cond_tensor).all().item()}"
        )

        #new_add: verify conditioning channel number
        if hasattr(self.model, "cond_channels"):

            if cond_tensor.shape[1] != self.model.cond_channels:

                raise ValueError(
                    f"Conditioning channel mismatch: "
                    f"model expects {self.model.cond_channels}, "
                    f"but received {cond_tensor.shape[1]}"
                )

        #new_add: verify conditioning has model spatial dimensions
        if self.image_size is not None:

            if (
                cond_tensor.shape[-2] != self.image_size
                or cond_tensor.shape[-1] != self.image_size
            ):

                raise ValueError(
                    f"Conditioning tensor has spatial size "
                    f"{cond_tensor.shape[-2]}x"
                    f"{cond_tensor.shape[-1]}, "
                    f"but model image_size is "
                    f"{self.image_size}"
                )

        if not th.isfinite(cond_tensor).all():

            raise ValueError(
                "Non-finite conditioning values detected "
                "before diffusion sampling"
            )
        
        # Create initial noise
        shape = (
            num_samples,
            self.model.in_channels,
            cond_tensor.shape[2],
            cond_tensor.shape[3]
        )

        noise = th.randn(
            shape,
            device=self.device
        )
        
        print(
            f"Initial noise shape: "
            f"{shape}"
        )

        print(
            f"Sampling with "
            f"{self.diffusion.num_timesteps} steps..."
        )
        
        # Time the sampling
        start_time = time.time()
        
        # Run sampling
        samples = self.diffusion.p_sample_loop(
            self.model,
            shape,
            noise=noise,
            clip_denoised=True,
            model_kwargs={
                "cond": cond_tensor
            },
            device=self.device,
            progress=progress
        )
        
        sampling_time = (
            time.time()
            - start_time
        )
        
        print(
            f"Generated padded samples shape: "
            f"{samples.shape}"
        )

        #new_add: verify generated padded size
        if self.image_size is not None:

            if (
                samples.shape[-2] != self.image_size
                or samples.shape[-1] != self.image_size
            ):

                raise ValueError(
                    f"Generated sample size is "
                    f"{samples.shape[-2]}x{samples.shape[-1]}, "
                    f"expected "
                    f"{self.image_size}x{self.image_size}"
                )

        #new_add: crop model output back to original geographical size
        if (
            self.original_height is not None
            and self.original_width is not None
        ):

            sample_h = samples.shape[-2]
            sample_w = samples.shape[-1]

            if (
                sample_h != self.original_height
                or sample_w != self.original_width
            ):

                crop_h = (
                    sample_h
                    - self.original_height
                )

                crop_w = (
                    sample_w
                    - self.original_width
                )

                if crop_h < 0 or crop_w < 0:

                    raise ValueError(
                        f"Generated sample size "
                        f"{sample_h}x{sample_w} "
                        f"is smaller than original grid "
                        f"{self.original_height}x"
                        f"{self.original_width}"
                    )

                crop_top = (
                    crop_h
                    // 2
                )

                crop_left = (
                    crop_w
                    // 2
                )

                samples = samples[
                    ...,
                    crop_top:
                        crop_top
                        + self.original_height,
                    crop_left:
                        crop_left
                        + self.original_width
                ]

                print(
                    f"Cropped samples back to original size: "
                    f"{samples.shape}"
                )

        #new_add: final shape verification
        if (
            self.original_height is not None
            and self.original_width is not None
        ):

            if (
                samples.shape[-2] != self.original_height
                or samples.shape[-1] != self.original_width
            ):

                raise ValueError(
                    f"Final sample shape "
                    f"{samples.shape[-2]}x{samples.shape[-1]} "
                    f"does not match original grid "
                    f"{self.original_height}x"
                    f"{self.original_width}"
                )

        print(
            f"Generated samples shape: "
            f"{samples.shape}"
        )

        print(
            f"Generated samples finite: "
            f"{th.isfinite(samples).all().item()}"
        )

        print(
            f"Sampling time: "
            f"{sampling_time:.2f} seconds"
        )
        
        return samples, sampling_time
    
    def denormalize_samples(self, samples):
        """Convert normalized samples back to original scale"""

        if self.target_normalizer:

            # Ensure normalizer is on same device
            if (
                self.target_normalizer.min.device
                != samples.device
            ):

                self.target_normalizer.min = (
                    self.target_normalizer.min.to(
                        samples.device
                    )
                )

                self.target_normalizer.max = (
                    self.target_normalizer.max.to(
                        samples.device
                    )
                )

                self.target_normalizer.range = (
                    self.target_normalizer.range.to(
                        samples.device
                    )
                )

            samples_denorm = (
                self.target_normalizer
                .denormalize(
                    samples
                )
            )

            print(
                f"Denormalized samples shape: "
                f"{samples_denorm.shape}"
            )

            print(
                f"Denormalized samples finite: "
                f"{th.isfinite(samples_denorm).all().item()}"
            )

            return samples_denorm

        return samples
    
    def get_model_info(self):
        """Get model information dictionary"""

        if self.model is None:
            return {}
        
        total_params = sum(
            p.numel()
            for p in self.model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "in_channels": self.model.in_channels,
            "out_channels": self.model.out_channels,
            "model_channels": self.model.model_channels,
            "diffusion_steps": (
                self.diffusion.num_timesteps
                if self.diffusion
                else "N/A"
            ),
        }
