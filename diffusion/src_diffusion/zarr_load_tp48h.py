from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
import zarr
import torch
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os


class DateRangeSplitter:
    def __init__(
        self,
        train_range: Tuple[str, str],
        val_range: Tuple[str, str],
        test_range: Tuple[str, str],
        zarr_path: Optional[str] = None,
        zarr_mode: str = "r",
        zarr_group: Optional[str] = None,
        time_key: str = "valid_time"
    ):
        self.ranges = {
            "train": train_range,
            "val": val_range,
            "test": test_range
        }

        self.time_key = time_key
        self._validate_ranges()

        self.root = None

        if zarr_path:
            self.root = zarr.open(
                zarr_path,
                mode=zarr_mode
            )

            if zarr_group:
                self.root = self.root[zarr_group]


    def _validate_ranges(self):

        for split, (start, end) in self.ranges.items():

            if pd.Timestamp(start) > pd.Timestamp(end):

                raise ValueError(
                    f"Invalid {split} range: "
                    f"{start} > {end}"
                )


    def _load_timestamps_from_zarr(
        self
    ) -> pd.DatetimeIndex:

        if self.root is None:

            raise ValueError(
                "Zarr root not initialized. "
                "Provide zarr_path."
            )

        if self.time_key not in self.root:

            raise KeyError(
                f"'{self.time_key}' not found in "
                f"{list(self.root.keys())}"
            )

        time_arr = self.root[self.time_key]

        raw_values = time_arr[:]

        units_str = time_arr.attrs.get(
            'units',
            'seconds since 1970-01-01'
        )

        unit_map = {
            'seconds': 's',
            'minutes': 'm',
            'hours': 'h',
            'days': 'D'
        }

        unit_key = (
            units_str
            .split(' ')[0]
            .lower()
        )

        pd_unit = unit_map.get(
            unit_key,
            's'
        )

        return pd.to_datetime(
            raw_values,
            unit=pd_unit,
            origin='unix',
            utc=True
        )


    def get_splits(
        self,
        timestamps: Optional[
            Union[
                pd.DatetimeIndex,
                np.ndarray
            ]
        ] = None
    ) -> Dict[str, List[int]]:

        if timestamps is None:

            timestamps = (
                self._load_timestamps_from_zarr()
            )

        if not isinstance(
            timestamps,
            pd.DatetimeIndex
        ):

            timestamps = pd.to_datetime(
                timestamps,
                unit='s',
                utc=True
            )

        elif timestamps.tz is None:

            timestamps = (
                timestamps.tz_localize('UTC')
            )

        results = {}

        for (
            split_name,
            (start_str, end_str)
        ) in self.ranges.items():

            start_ts = pd.Timestamp(
                start_str,
                tz='UTC'
            )

            end_ts = pd.Timestamp(
                end_str,
                tz='UTC'
            )

            mask = (
                (timestamps >= start_ts)
                &
                (timestamps <= end_ts)
            )

            indices = np.where(
                mask
            )[0]

            results[split_name] = (
                indices.tolist()
            )

            print(
                f" {split_name.capitalize()}: "
                f"{len(indices)} samples found."
            )

        return results


class ZarrNormalizer:

    def __init__(
        self,
        min_val: torch.Tensor,
        max_val: torch.Tensor
    ):

        self.min = min_val.view(
            -1,
            1,
            1
        )

        self.max = max_val.view(
            -1,
            1,
            1
        )

        self.range = (
            self.max - self.min
        )

        self.range[
            self.range == 0
        ] = 1.0


    def normalize(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:

        return (
            2.0
            * (tensor - self.min)
            / self.range
            - 1.0
        )


    def denormalize(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:

        return (
            ((tensor + 1.0) / 2.0)
            * self.range
            + self.min
        )


class ZarrWeatherDataset(Dataset):

    def __init__(
        self,
        zarr_path: str,
        indices: list,
        input_variables: list,
        target_variables: list,
        input_normalizer: Optional = None,
        target_normalizer: Optional = None,
        # image_size: int = 256,
        image_size: int = 1024,
        add_seasonal_cond: bool = False,
        seasonal_channels: int = 0,
        forecast_lead_hours: int = 0,
    ):

        self.store = zarr.open(
            zarr_path,
            mode='r'
        )

        self.indices = indices

        self.input_variables = (
            input_variables
        )

        self.target_variables = (
            target_variables
        )

        self.input_normalizer = (
            input_normalizer
        )

        self.target_normalizer = (
            target_normalizer
        )

        self.image_size = image_size

        self.add_seasonal_cond = (
            add_seasonal_cond
        )

        self.seasonal_channels = (
            seasonal_channels
        )

        self.forecast_lead_hours = int(
            forecast_lead_hours
        )

        # =========================================================
        # Load valid times and build forecast input-target pairs
        #
        # For a 24-hour forecast:
        #
        # input  = atmospheric/land conditioning at time t
        # target = swvl1/stl1 at time t + 24 h
        #
        # A pair is retained only when BOTH initialization and
        # target times belong to the supplied split indices.
        # This prevents leakage across train/validation/test splits.
        # =========================================================

        time_arr = (
            self.store['valid_time']
        )

        raw_times = time_arr[:]

        # Confirmed Unix-second valid_time
        self.times = pd.to_datetime(
            raw_times,
            unit='s',
            origin='unix',
            utc=True
        )

        allowed_indices = set(
            int(i)
            for i in self.indices
        )

        time_to_index = {
            timestamp: i
            for i, timestamp in enumerate(
                self.times
            )
        }

        lead_delta = pd.Timedelta(
            hours=self.forecast_lead_hours
        )

        self.sample_pairs = []

        for input_idx in self.indices:

            input_idx = int(
                input_idx
            )

            target_time = (
                self.times[input_idx]
                + lead_delta
            )

            target_idx = time_to_index.get(
                target_time
            )

            if target_idx is None:
                continue

            if target_idx not in allowed_indices:
                continue

            if "tp" in self.input_variables:

                precipitation_time = (
                    target_time
                    - pd.Timedelta(
                        hours=48
                    )
                )

                precipitation_idx = (
                    time_to_index.get(
                        precipitation_time
                    )
                )

                if precipitation_idx is None:
                    continue

                if precipitation_idx not in allowed_indices:
                    continue

            self.sample_pairs.append(
                (
                    input_idx,
                    int(target_idx)
                )
            )

        print(
            f"Forecast lead: "
            f"{self.forecast_lead_hours} hours"
        )

        print(
            f"Valid forecast pairs: "
            f"{len(self.sample_pairs)}"
        )

        if len(self.sample_pairs) == 0:

            raise ValueError(
                f"No valid input-target pairs found for "
                f"forecast lead {self.forecast_lead_hours} hours. "
                f"Check valid_time coverage and split ranges."
            )


    def __len__(self):

        return len(
            self.sample_pairs
        )


    def __getitem__(
        self,
        idx
    ):

        input_idx, target_idx = (
            self.sample_pairs[idx]
        )

        # =========================================================
        # Load conditioning/input variables
        # =========================================================

        input_data_list = []

        for var in self.input_variables:

            if var == "tp":

                precipitation_time = (
                    self.times[target_idx]
                    - pd.Timedelta(
                        hours=48
                    )
                )

                precipitation_matches = np.where(
                    self.times
                    == precipitation_time
                )[0]

                if len(
                    precipitation_matches
                ) == 0:

                    raise ValueError(
                        f"Precipitation time "
                        f"{precipitation_time} "
                        f"not found in Zarr dataset"
                    )

                precipitation_idx = int(
                    precipitation_matches[0]
                )

                data = (
                    self.store[var][
                        precipitation_idx
                    ]
                    .astype(np.float32)
                )

            else:

                data = (
                    self.store[var][input_idx]
                    .astype(np.float32)
                )

            input_data_list.append(
                data
            )


        # =========================================================
        # Load target variables
        # =========================================================

        target_data_list = []

        for var in self.target_variables:

            data = (
                self.store[var][target_idx]
                .astype(np.float32)
            )

            target_data_list.append(
                data
            )


        # =========================================================
        # Convert to torch tensors
        # =========================================================

        input_tensor = torch.from_numpy(
            np.stack(
                input_data_list,
                axis=0
            )
        ).float()

        target_tensor = torch.from_numpy(
            np.stack(
                target_data_list,
                axis=0
            )
        ).float()


        # =========================================================
        # Normalize
        # =========================================================

        if self.input_normalizer:

            input_tensor = (
                self.input_normalizer
                .normalize(
                    input_tensor
                )
            )


        if self.target_normalizer:

            target_tensor = (
                self.target_normalizer
                .normalize(
                    target_tensor
                )
            )


        # =========================================================
        # Handle NaN and Inf values
        #
        # IMPORTANT:
        # Normalized data are expected approximately in [-1, 1].
        #
        # NaN    ->  0.0
        # +Inf   -> +1.0
        # -Inf   -> -1.0
        #
        # This prevents complete batches from being skipped in
        # diffusion_train.py because torch.isfinite() fails.
        # =========================================================

        input_tensor = torch.nan_to_num(
            input_tensor,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        target_tensor = torch.nan_to_num(
            target_tensor,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )


        # =========================================================
        # Add seasonal conditioning
        # =========================================================

        if (
            self.add_seasonal_cond
            and self.seasonal_channels > 0
        ):

            h = input_tensor.shape[-2]
            w = input_tensor.shape[-1]

            timestamp = (
                self.times[target_idx]
            )

            day_of_year = (
                timestamp.dayofyear
            )

            seasonal_features = []

            frequencies = [
                1,
                2,
                3,
                4,
                6,
                12
            ]

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


            seasonal_features = (
                seasonal_features[
                    :self.seasonal_channels
                ]
            )


            seasonal_tensor = (
                torch.tensor(
                    seasonal_features,
                    dtype=torch.float32
                )
            )


            seasonal_expanded = (
                seasonal_tensor
                .view(
                    -1,
                    1,
                    1
                )
                .expand(
                    -1,
                    h,
                    w
                )
            )


            input_tensor = torch.cat(
                [
                    input_tensor,
                    seasonal_expanded
                ],
                dim=0
            )


        # =========================================================
        # Pad spatial dimensions to image_size
        #
        # Example:
        #
        # Original:
        #       341 x 341
        #
        # Model:
        #       352 x 352
        #
        # Padding:
        #
        # left   = 5
        # right  = 6
        # top    = 5
        # bottom = 6
        #
        # =========================================================

        h = target_tensor.shape[-2]
        w = target_tensor.shape[-1]

        pad_h = (
            self.image_size - h
        )

        pad_w = (
            self.image_size - w
        )


        if (
            pad_h < 0
            or pad_w < 0
        ):

            raise ValueError(
                f"Data size {h}x{w} "
                f"is larger than requested "
                f"image_size {self.image_size}"
            )


        if (
            pad_h > 0
            or pad_w > 0
        ):

            pad_top = (
                pad_h // 2
            )

            pad_bottom = (
                pad_h
                - pad_top
            )

            pad_left = (
                pad_w // 2
            )

            pad_right = (
                pad_w
                - pad_left
            )


            # -----------------------------------------------------
            # Pad TARGET
            # -----------------------------------------------------

            target_tensor = F.pad(
                target_tensor,
                (
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom
                ),
                mode="replicate"
            )


            # -----------------------------------------------------
            # Pad CONDITION
            # -----------------------------------------------------

            input_tensor = F.pad(
                input_tensor,
                (
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom
                ),
                mode="replicate"
            )


        # =========================================================
        # Final safety check
        # =========================================================

        if not torch.isfinite(
            target_tensor
        ).all():

            raise ValueError(
                f"Non-finite values remain in TARGET "
                f"after cleaning at target dataset index "
                f"{target_idx}"
            )


        if not torch.isfinite(
            input_tensor
        ).all():

            raise ValueError(
                f"Non-finite values remain in CONDITION "
                f"after cleaning at input dataset index "
                f"{input_idx}"
            )


        # =========================================================
        # Return:
        #
        # target_tensor = diffusion target
        # input_tensor  = conditioning fields
        # =========================================================

        return (
            target_tensor,
            input_tensor
        )


def get_vars_indices(
    target_vars,
    z
):

    vars_indices = []

    all_variables = np.array(
        z['variable']
    )


    for i in range(
        len(target_vars)
    ):

        for j in range(
            len(all_variables)
        ):

            if (
                target_vars[i]
                == all_variables[j]
            ):

                vars_indices.append(
                    j
                )

                break


    return vars_indices


def load_data(
    zarr_path: str,
    batch_size: int,
    # image_size: int = 256,
    image_size: int = 1024,

    train_range: Tuple[str, str] = (
        "2010-01-01",
        "2011-12-31"
    ),

    val_range: Tuple[str, str] = (
        "2020-01-01",
        "2020-08-15"
    ),

    test_range: Tuple[str, str] = (
        "2018-08-16",
        "2019-12-31"
    ),

    input_vars: List[str] = None,
    target_vars: List[str] = None,

    num_workers: int = 2,

    shuffle: bool = True,

    pin_memory: bool = True,

    drop_last: bool = True,

    normalize: bool = True,

    class_cond: bool = False,

    deterministic: bool = False,

    add_seasonal_cond: bool = False,

    seasonal_channels: int = 0,

    forecast_lead_hours: int = 0,
):

    print("=" * 80)

    print(
        "Loading Zarr Data for "
        "Diffusion Training"
    )

    print("=" * 80)


    # =============================================================
    # Default variables
    # =============================================================

    if input_vars is None:

        input_vars = [
            't2m_era5_mean',
            't2m_era5_std',
            # 'd2m_era5_mean',
            # 'sp_era5_mean',
        ]


    if target_vars is None:

        target_vars = [
            't2m_cerra_mean',
            't2m_cerra_std'
        ]


    # =============================================================
    # Step 1: Date-based split
    # =============================================================

    print(
        "\n--- Step 1: "
        "Creating Date-Based Splits ---"
    )


    splitter = DateRangeSplitter(
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        zarr_path=zarr_path
    )


    splits = (
        splitter.get_splits()
    )


    train_indices = (
        splits['train']
    )


    print(
        f"Training samples: "
        f"{len(train_indices)}"
    )


    # =============================================================
    # Step 2: Statistics
    # =============================================================

    print(
        "\n--- Step 2: "
        "Loading Statistics for "
        "Normalization ---"
    )


    z = zarr.open(
        zarr_path,
        mode='r'
    )


    target_vars_indices = (
        get_vars_indices(
            target_vars,
            z
        )
    )


    input_vars_indices = (
        get_vars_indices(
            input_vars,
            z
        )
    )


    min_stats = torch.from_numpy(
        z['min'][:]
        .astype(np.float32)
    )


    max_stats = torch.from_numpy(
        z['max'][:]
        .astype(np.float32)
    )


    target_min_stats = (
        min_stats[
            target_vars_indices
        ]
    )


    target_max_stats = (
        max_stats[
            target_vars_indices
        ]
    )


    input_min_stats = (
        min_stats[
            input_vars_indices
        ]
    )


    input_max_stats = (
        max_stats[
            input_vars_indices
        ]
    )


    print(
        f"Input variables: "
        f"{input_vars}"
    )


    print(
        f"Target variables: "
        f"{target_vars}"
    )


    # =============================================================
    # Step 3: Normalizers
    # =============================================================

    print(
        "\n--- Step 3: "
        "Initializing Normalizers ---"
    )


    target_normalizer = (

        ZarrNormalizer(
            min_val=target_min_stats,
            max_val=target_max_stats
        )

        if normalize

        else None
    )


    input_normalizer = (

        ZarrNormalizer(
            min_val=input_min_stats,
            max_val=input_max_stats
        )

        if normalize

        else None
    )


    # =============================================================
    # Step 4: Dataset
    # =============================================================

    print(
        "\n--- Step 4: "
        "Creating Dataset ---"
    )


    print(
        f"Forecast lead hours: "
        f"{forecast_lead_hours}"
    )


    dataset = ZarrWeatherDataset(
        zarr_path=zarr_path,
        indices=train_indices,
        input_variables=input_vars,
        target_variables=target_vars,
        input_normalizer=input_normalizer,
        target_normalizer=target_normalizer,
        image_size=image_size,
        add_seasonal_cond=add_seasonal_cond,
        seasonal_channels=seasonal_channels,
        forecast_lead_hours=forecast_lead_hours,
    )


    print(
        f"Dataset created with "
        f"{len(dataset)} samples"
    )


    # =============================================================
    # Verify first sample
    # =============================================================

    sample_target, sample_condition = (
        dataset[0]
    )


    print(
        f"Sample shape: "
        f"{sample_target.shape} "
        f"(target), "
        f"{sample_condition.shape} "
        f"(condition)"
    )


    print(
        f"Target finite: "
        f"{torch.isfinite(sample_target).all().item()}"
    )


    print(
        f"Condition finite: "
        f"{torch.isfinite(sample_condition).all().item()}"
    )


    # =============================================================
    # Step 5: DataLoader
    # =============================================================

    print(
        "\n--- Step 5: "
        "Creating DataLoader ---"
    )


    print(
        f"DataLoader num_workers = "
        f"{num_workers}"
    )


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


    print(
        f"DataLoader created with "
        f"batch size {batch_size}"
    )


    print("=" * 80)

    return loader
#--------------Debuging PLOTS------------------
##
def plot_actual_vs_normalized(
    zarr_path: str,
    input_vars: List[str],
    target_vars: List[str],
    verification_time: str,
    output_dir: str = "normalization_verification",
    forecast_lead_hours: int = 0,
):
    """
    Plot actual and normalized geographical fields for all physical
    conditioning and target variables at one specified date/time.

    Parameters
    ----------
    zarr_path : str
        Path to Zarr dataset.

    input_vars : list
        Physical conditioning variables.

    target_vars : list
        Target variables.

    verification_time : str
        Date/time to plot, for example:
            "2010-01-15 12:00:00"

    output_dir : str
        Directory where PNG figures are written.

    forecast_lead_hours : int
        Forecast lead in hours. Input variables are plotted at
        verification_time and target variables at verification_time
        + forecast_lead_hours.
    """

    print("=" * 80)
    print("NORMALIZATION VERIFICATION")
    print("=" * 80)

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Open Zarr
    # ---------------------------------------------------------

    z = zarr.open(
        zarr_path,
        mode="r"
    )

    # ---------------------------------------------------------
    # Read time coordinate
    # ---------------------------------------------------------

    if "valid_time" not in z:
        raise KeyError(
            "'valid_time' not found in Zarr dataset"
        )

    raw_times = z["valid_time"][:]

    times = pd.to_datetime(
        raw_times,
        unit="s",
        origin="unix",
        utc=True
    )

    requested_time = pd.Timestamp(
        verification_time
    )

    if requested_time.tzinfo is None:
        requested_time = requested_time.tz_localize(
            "UTC"
        )
    else:
        requested_time = requested_time.tz_convert(
            "UTC"
        )

    # ---------------------------------------------------------
    # Find exact requested time
    # ---------------------------------------------------------

    matches = np.where(
        times == requested_time
    )[0]

    if len(matches) == 0:

        # Find nearest time for useful diagnostic
        time_difference = np.abs(
            times - requested_time
        )

        nearest_index = np.argmin(
            time_difference
        )

        nearest_time = times[
            nearest_index
        ]

        raise ValueError(
            f"Requested time {requested_time} "
            f"was not found.\n"
            f"Nearest available time is {nearest_time}"
        )

    time_index = int(
        matches[0]
    )

    target_time = (
        requested_time
        + pd.Timedelta(
            hours=forecast_lead_hours
        )
    )

    target_matches = np.where(
        times == target_time
    )[0]

    if len(target_matches) == 0:

        raise ValueError(
            f"Forecast target time {target_time} "
            f"was not found in the Zarr dataset."
        )

    target_time_index = int(
        target_matches[0]
    )

    print(
        f"Requested time : {requested_time}"
    )

    print(
        f"Zarr index     : {time_index}"
    )

    print(
        f"Forecast lead  : {forecast_lead_hours} hours"
    )

    print(
        f"Target time    : {target_time}"
    )

    print(
        f"Target index   : {target_time_index}"
    )

    # ---------------------------------------------------------
    # Read geographical coordinates
    # ---------------------------------------------------------

    if "latitude" not in z:
        raise KeyError(
            "'latitude' not found in Zarr dataset"
        )

    if "longitude" not in z:
        raise KeyError(
            "'longitude' not found in Zarr dataset"
        )

    latitude = np.asarray(
        z["latitude"][:]
    )

    longitude = np.asarray(
        z["longitude"][:]
    )

    print(
        f"Latitude shape  : {latitude.shape}"
    )

    print(
        f"Longitude shape : {longitude.shape}"
    )

    # ---------------------------------------------------------
    # Variable metadata
    # ---------------------------------------------------------

    all_variables = np.array(
        z["variable"]
    )

    min_stats = np.asarray(
        z["min"][:],
        dtype=np.float32
    )

    max_stats = np.asarray(
        z["max"][:],
        dtype=np.float32
    )

    # ---------------------------------------------------------
    # Combine physical input and target variables
    # ---------------------------------------------------------

    variables_to_plot = []

    for var in input_vars:

        if var not in variables_to_plot:
            variables_to_plot.append(
                var
            )

    for var in target_vars:

        if var not in variables_to_plot:
            variables_to_plot.append(
                var
            )

    print(
        f"Variables to plot: "
        f"{variables_to_plot}"
    )

    # ---------------------------------------------------------
    # Process each variable
    # ---------------------------------------------------------

    for var in variables_to_plot:

        print(
            "\n" + "-" * 80
        )

        print(
            f"Processing variable: {var}"
        )

        # -----------------------------------------------------
        # Check variable exists
        # -----------------------------------------------------

        if var not in z:

            print(
                f"WARNING: {var} not found "
                f"in Zarr. Skipping."
            )

            continue

        # -----------------------------------------------------
        # Find normalization-statistic index
        # -----------------------------------------------------

        variable_matches = np.where(
            all_variables == var
        )[0]

        if len(variable_matches) == 0:

            print(
                f"WARNING: {var} not found "
                f"in z['variable']. Skipping."
            )

            continue

        variable_index = int(
            variable_matches[0]
        )

        min_value = float(
            min_stats[variable_index]
        )

        max_value = float(
            max_stats[variable_index]
        )

        value_range = (
            max_value - min_value
        )

        if value_range == 0:

            value_range = 1.0

        # -----------------------------------------------------
        # Read actual field
        # -----------------------------------------------------

        if var in target_vars:

            field_time_index = (
                target_time_index
            )

            field_time = (
                target_time
            )

        elif var == "tp":

            precipitation_time = (
                target_time
                - pd.Timedelta(
                    hours=48
                )
            )

            precipitation_matches = np.where(
                times
                == precipitation_time
            )[0]

            if len(
                precipitation_matches
            ) == 0:

                raise ValueError(
                    f"Precipitation time "
                    f"{precipitation_time} "
                    f"was not found in the Zarr dataset."
                )

            field_time_index = int(
                precipitation_matches[0]
            )

            field_time = (
                precipitation_time
            )

        else:

            field_time_index = (
                time_index
            )

            field_time = (
                requested_time
            )

        actual = np.asarray(
            z[var][field_time_index],
            dtype=np.float32
        )

        print(
            f"Actual shape : {actual.shape}"
        )

        print(
            f"Normalization min = {min_value}"
        )

        print(
            f"Normalization max = {max_value}"
        )

        # -----------------------------------------------------
        # Normalize exactly like ZarrNormalizer
        #
        # normalized =
        #
        #      2 * (actual - min)
        #      ------------------ - 1
        #          max - min
        #
        # -----------------------------------------------------

        normalized = (
            2.0
            * (
                actual - min_value
            )
            / value_range
            - 1.0
        )

        # -----------------------------------------------------
        # Count invalid values BEFORE replacement
        # -----------------------------------------------------

        actual_nan = int(
            np.isnan(actual).sum()
        )

        actual_inf = int(
            np.isinf(actual).sum()
        )

        normalized_nan = int(
            np.isnan(normalized).sum()
        )

        normalized_inf = int(
            np.isinf(normalized).sum()
        )

        print(
            f"Actual NaN       : {actual_nan}"
        )

        print(
            f"Actual Inf       : {actual_inf}"
        )

        print(
            f"Normalized NaN   : {normalized_nan}"
        )

        print(
            f"Normalized Inf   : {normalized_inf}"
        )

        # -----------------------------------------------------
        # Keep copies containing NaNs for plotting.
        #
        # This is important because we want the verification
        # plots to show where missing data occur.
        # -----------------------------------------------------

        actual_plot = actual.copy()

        normalized_plot = (
            normalized.copy()
        )

        # -----------------------------------------------------
        # Clean normalized field exactly as training loader does
        # -----------------------------------------------------

        normalized_clean = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        # -----------------------------------------------------
        # Print statistics
        # -----------------------------------------------------

        finite_actual = actual[
            np.isfinite(actual)
        ]

        finite_normalized = normalized[
            np.isfinite(normalized)
        ]

        if finite_actual.size > 0:

            actual_min = np.min(
                finite_actual
            )

            actual_max = np.max(
                finite_actual
            )

            actual_mean = np.mean(
                finite_actual
            )

            print(
                f"Actual finite range : "
                f"{actual_min:.6g} "
                f"to {actual_max:.6g}"
            )

            print(
                f"Actual finite mean  : "
                f"{actual_mean:.6g}"
            )

        if finite_normalized.size > 0:

            norm_min = np.min(
                finite_normalized
            )

            norm_max = np.max(
                finite_normalized
            )

            norm_mean = np.mean(
                finite_normalized
            )

            print(
                f"Normalized range    : "
                f"{norm_min:.6g} "
                f"to {norm_max:.6g}"
            )

            print(
                f"Normalized mean     : "
                f"{norm_mean:.6g}"
            )

        print(
            f"Clean normalized range : "
            f"{normalized_clean.min():.6g} "
            f"to {normalized_clean.max():.6g}"
        )

        # -----------------------------------------------------
        # Create figure
        # -----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(16, 7)
        )

        # -----------------------------------------------------
        # Actual field
        # -----------------------------------------------------

        if (
            latitude.ndim == 2
            and longitude.ndim == 2
            and latitude.shape == actual_plot.shape
            and longitude.shape == actual_plot.shape
        ):

            plot1 = axes[0].pcolormesh(
                longitude,
                latitude,
                actual_plot,
                shading="auto"
            )

        elif (
            latitude.ndim == 1
            and longitude.ndim == 1
        ):

            plot1 = axes[0].pcolormesh(
                longitude,
                latitude,
                actual_plot,
                shading="auto"
            )

        else:

            plot1 = axes[0].imshow(
                actual_plot,
                origin="lower",
                aspect="auto"
            )

        axes[0].set_title(
            f"{var}: Actual value"
        )

        axes[0].set_xlabel(
            "Longitude"
        )

        axes[0].set_ylabel(
            "Latitude"
        )

        cbar1 = fig.colorbar(
            plot1,
            ax=axes[0],
            orientation="vertical"
        )

        cbar1.set_label(
            "Actual value"
        )

        # -----------------------------------------------------
        # Normalized field
        #
        # Use fixed -1 to +1 range so every normalized plot
        # can be interpreted consistently.
        # -----------------------------------------------------

        if (
            latitude.ndim == 2
            and longitude.ndim == 2
            and latitude.shape == normalized_plot.shape
            and longitude.shape == normalized_plot.shape
        ):

            plot2 = axes[1].pcolormesh(
                longitude,
                latitude,
                normalized_plot,
                shading="auto",
                vmin=-1.0,
                vmax=1.0
            )

        elif (
            latitude.ndim == 1
            and longitude.ndim == 1
        ):

            plot2 = axes[1].pcolormesh(
                longitude,
                latitude,
                normalized_plot,
                shading="auto",
                vmin=-1.0,
                vmax=1.0
            )

        else:

            plot2 = axes[1].imshow(
                normalized_plot,
                origin="lower",
                aspect="auto",
                vmin=-1.0,
                vmax=1.0
            )

        axes[1].set_title(
            f"{var}: Normalized value"
        )

        axes[1].set_xlabel(
            "Longitude"
        )

        axes[1].set_ylabel(
            "Latitude"
        )

        cbar2 = fig.colorbar(
            plot2,
            ax=axes[1],
            orientation="vertical"
        )

        cbar2.set_label(
            "Normalized value"
        )

        # -----------------------------------------------------
        # Main title
        # -----------------------------------------------------

        fig.suptitle(
            f"{var} | {field_time} | "
            f"Normalization min={min_value:.5g}, "
            f"max={max_value:.5g}",
            fontsize=14
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        safe_time = (
            requested_time
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        output_file = os.path.join(
            output_dir,
            f"{var}_actual_normalized_"
            f"{safe_time}.png"
        )

        fig.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(
            fig
        )

        print(
            f"Saved: {output_file}"
        )

    print("\n" + "=" * 80)

    print(
        "NORMALIZATION VERIFICATION COMPLETE"
    )

    print(
        f"Figures written to: {output_dir}"
    )

    print("=" * 80)
