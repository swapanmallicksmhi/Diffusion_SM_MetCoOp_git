import os
import xarray as xr
from helper_era5land import find_matching_files, crop_era5land
from helper_era5land import plot_all, save_to_zarr, calculate_summary_stats
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run era-land processing")

    parser.add_argument("--grid_size", type=int, required=True,
                        help="Size of the grid, 341 or 256")

    parser.add_argument("--era5land_dir", type=str, required=True,
                        help="Path to ERA5-Land data directory")

    parser.add_argument("--output_dir", type=str, default="../output_plots",
                        help="Output directory for plots")

    parser.add_argument("--zarr_path", type=str, default="../output_data.zarr",
                        help="Output Zarr store")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    GRID_SIZE = args.grid_size
    ERA_DIR = args.era5land_dir
    OUTPUT_DIR = args.output_dir
    zarr_path = args.zarr_path

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    YEARS = [str(i) for i in range(2016, 2021)]
    MONTHS = [f"{i:02d}" for i in range(1, 13)]
    CYCLES = ['0000', '0600', '1200', '1800']

    LAT_MIN, LAT_MAX = 50, 73
    LON_MIN, LON_MAX = 1, 25


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

    pairs = find_matching_files(ERA_DIR, YEARS, MONTHS, CYCLES)

    if not pairs:
        print("No matching files found.")
    else:
        for era5land_path in pairs:

            print(f"\nProcessing: {os.path.basename(era5land_path)}")

            try:
                ds_c = xr.open_dataset(era5land_path)

                ds_c['longitude'] = ((ds_c.longitude + 180) % 360) - 180

                if 'expver' in ds_c.coords or 'expver' in ds_c.data_vars:
                    ds_c = ds_c.drop_vars('expver')

                available_vars = [v for v in variables if v in ds_c]

                if not available_vars:
                    raise ValueError("No requested ERA5-Land variables found in dataset")

                ds_c = ds_c[available_vars]

                ds_c_sub = crop_era5land(
                    ds_c,
                    LAT_MIN, LAT_MAX,
                    LON_MIN, LON_MAX,
                    GRID_SIZE
                )

                n_times = ds_c_sub.sizes['valid_time']

                for t in range(n_times):

                    time_val = ds_c_sub.valid_time.isel(valid_time=t).values
                    time_str_file = str(time_val)[:16].replace(':', '').replace('-', '').replace('T', '_')

                    idx = {'valid_time': t}

                    plot_title = f"(Time: {str(time_val)[:16]})"

                    for var in available_vars:

                        filename = f"plot_{var}_{time_str_file}.png"
                        save_full_path = os.path.join(OUTPUT_DIR, filename)

                        #plot_all(
                        #    ds_c_sub,
                        #    idx,
                        #    varname=var,
                        #    title_suffix=plot_title,
                        #    save_path=save_full_path
                        #)

                save_to_zarr(zarr_path=zarr_path, ds_era5land=ds_c_sub)
                print(zarr_path)

                ds_c.close()

            except Exception as e:
                print(f"Error processing {os.path.basename(era5land_path)}: {e}")

        calculate_summary_stats(zarr_path)
