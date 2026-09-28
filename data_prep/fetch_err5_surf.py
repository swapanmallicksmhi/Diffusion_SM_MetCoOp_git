import cdsapi
import calendar
from pathlib import Path

dataset = "reanalysis-era5-land"
output_dir = Path("/lus/h2resw01/scratch/swe4281/CERRA_DATA2026/ERA5EDA_DATA/SURF_METCOOP")
output_dir.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

times = ["00:00", "06:00", "12:00", "18:00"]

for year in range(1985, 2025):
    year_str = str(year)

    for time in times:
        for month in range(1, 13):

            mm = f"{month:02d}"
            ndays = calendar.monthrange(year, month)[1]

            request = {
                "variable": [
                    "2m_temperature",
                    "2m_dewpoint_temperature",
                    "skin_temperature",

                    "soil_temperature_level_1",
                    "soil_temperature_level_2",
                    "soil_temperature_level_3",
                    "soil_temperature_level_4",
                    "volumetric_soil_water_layer_1",
                    "volumetric_soil_water_layer_2",
                    "volumetric_soil_water_layer_3",
                    "volumetric_soil_water_layer_4",

                    "total_precipitation",
                    "surface_net_solar_radiation",
                    "surface_net_thermal_radiation",
                    "surface_latent_heat_flux",
                    "surface_sensible_heat_flux",

                    "potential_evaporation",
                    "total_evaporation",
                    "surface_runoff",
                    "sub_surface_runoff",

                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "surface_pressure",

                    "leaf_area_index_high_vegetation",
                    "leaf_area_index_low_vegetation"
                    "snow_albedo",
                    "snow_cover",
                    "snow_density",
                    "snow_depth",
                   #"soil_type",
                ],
                "year": year_str,
                "month": mm,
                "day": [f"{day:02d}" for day in range(1, ndays + 1)],
                "time": [time],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [73, 1, 50, 30]
            }
            outfile = output_dir / f"Metcoop_era5land_{year_str}{mm}_{time.replace(':','')}.nc"
            print(f"Downloading {outfile} ...", flush=True)
            client.retrieve(dataset, request, str(outfile))
            print(f"Completed: {outfile}", flush=True)
print("All years and months successfully downloaded.")
