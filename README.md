# Diffusion_Soil-Moisture

This repository provides a complete workflow for **high-resolution soil moisture and soil temperature prediction over the MetCoOp domain in the Nordic region** using a **conditional score-based diffusion model**. The framework uses **ERA5-Land** as the coarse-resolution conditioning dataset and a **high-resolution regional dataset over the MetCoOp domain** as the target.

The study domain covers the Nordic and surrounding northern European region represented by the MetCoOp numerical weather prediction domain, including major parts of **Sweden, Norway, Finland, Denmark, the Baltic region, and adjacent areas**.

## Scientific Background

Soil moisture and soil temperature are important land-surface variables controlling the terrestrial water and energy cycles, surface-atmosphere exchanges, evapotranspiration, runoff, and near-surface weather conditions.

Over the Nordic region, these variables are strongly influenced by pronounced seasonal variability, snow cover, frozen soil, vegetation, precipitation, and land-atmosphere interactions. Accurately representing their spatial variability is therefore important for high-resolution weather, hydrological, climate, and land-surface applications.

The model learns the nonlinear relationship between coarse-resolution **ERA5-Land forcing and land-surface variables** and the corresponding **high-resolution soil hydrothermal states over the MetCoOp domain**. The conditional diffusion framework is designed to reconstruct fine-scale spatial structures while also representing uncertainty through stochastic generation.

## Study Domain: MetCoOp / Nordic Region

The analysis is performed over the **MetCoOp domain**, covering the Nordic region and surrounding areas of northern Europe.

The geographical domain used during preprocessing, training, inference, and evaluation should be defined consistently in the corresponding configuration or preprocessing scripts using the required longitude and latitude boundaries.

Typical characteristics of the domain include:

- Northern European and Nordic land areas.
- Complex coastlines and numerous lakes.
- Mountainous terrain, particularly over Norway and western Scandinavia.
- Strong north-south climatic gradients.
- Seasonal snow and frozen-soil conditions.
- Large differences in vegetation and surface characteristics.

## Input Dataset: ERA5-Land

**ERA5-Land** is used as the coarse-resolution conditioning dataset. The selected variables provide information about near-surface atmospheric conditions, soil state, surface radiation, surface turbulent fluxes, hydrology, vegetation, and land-surface forcing.

```python
ERA5LAND_VARIABLES = [
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
    "leaf_area_index_low_vegetation",
    "snow information"
]
```

Before training, the ERA5-Land fields are spatially subset to the MetCoOp/Nordic domain and temporally matched with the corresponding high-resolution target data.

Depending on the model configuration, the input variables can be normalized and stored in an efficient format such as **Zarr** for training and inference.

## Target Dataset

The target dataset consists of high-resolution soil moisture and soil temperature fields covering the same MetCoOp domain and valid times as the ERA5-Land conditioning data.

```python
TARGET_VARIABLES = [
    "high_resolution_soil_moisture",
    "high_resolution_soil_temperature"
]
```

The high-resolution target data should be interpolated or remapped to a common training grid and processed using the same spatial domain and temporal sampling as the conditioning dataset.

The main target variables are:

- **Soil moisture**: high-resolution volumetric soil-water information.
- **Soil temperature**: high-resolution soil-temperature information.

Additional soil layers or land-surface variables can be included by extending the dataset configuration.

## Conditional Score-Based Diffusion Model

The score-based diffusion model learns the conditional probability distribution

\[
p(Y \mid X),
\]

where:

- \(X\) represents the ERA5-Land conditioning variables over the MetCoOp domain.
- \(Y\) represents the corresponding high-resolution soil moisture and soil temperature fields.

During the forward diffusion process, noise is progressively added to the target high-resolution fields. The neural network is trained to estimate the score of the perturbed conditional data distribution.

During inference, the reverse stochastic process starts from noise and progressively reconstructs high-resolution land-surface fields conditioned on the ERA5-Land input.

Because the reverse diffusion process is stochastic, multiple realizations can be generated for the same atmospheric and land-surface conditions. These realizations can be used to estimate **ensemble spread and uncertainty**.

## Data Preprocessing

The main preprocessing steps are:

1. Read ERA5-Land conditioning variables.
2. Select data covering the MetCoOp/Nordic domain.
3. Read the corresponding high-resolution target dataset.
4. Match conditioning and target data by valid time.
5. Regrid or interpolate the datasets when required.
6. Apply land/sea or other masks if required by the experiment.
7. Normalize the input and target variables.
8. Convert the processed datasets to the training format, for example **Zarr**.
9. Split the dataset into training, validation, and test periods.

All preprocessing parameters should remain consistent between training, validation, and inference.

## Training

During training, the model receives the coarse-resolution ERA5-Land variables as conditioning information and a noisy version of the high-resolution target field.

The model learns to recover the score required to reverse the diffusion process and reconstruct the high-resolution soil state.

The training procedure can include:

- Random diffusion timesteps.
- Different noise schedules.
- Seasonal or temporal conditioning.
- Multiple meteorological conditioning variables.
- Soil moisture and soil temperature as separate or joint prediction targets.
- Distributed GPU training for large datasets and model configurations.

## Inference

During inference, ERA5-Land fields corresponding to a selected valid time are used as conditioning input.

The trained diffusion model generates one or more high-resolution realizations of soil moisture and/or soil temperature over the MetCoOp domain.

Multiple samples may be generated to evaluate the probabilistic characteristics of the model and quantify uncertainty.

## Evaluation

Generated fields should be compared with the corresponding high-resolution reference data over the Nordic domain.

Possible deterministic evaluation metrics include:

- Root Mean Square Error (**RMSE**).
- Mean Absolute Error (**MAE**).
- Mean Bias.
- Spatial correlation.
- Temporal correlation.

Probabilistic evaluation can additionally include:

- Ensemble mean.
- Ensemble standard deviation.
- Spread-error relationship.
- Quantile-based diagnostics.
- Reliability diagnostics.

Evaluation can also be performed separately for different seasons and surface regimes because the Nordic land surface exhibits strong seasonal variability.

## Workflow

1. Download or prepare ERA5-Land data.
2. Prepare the high-resolution target dataset over the MetCoOp domain.
3. Select the common Nordic spatial domain.
4. Match ERA5-Land and target data in time.
5. Regrid and preprocess the datasets.
6. Normalize the input and target variables.
7. Convert the processed data to Zarr or the required training format.
8. Create training, validation, and test datasets.
9. Train the conditional score-based diffusion model.
10. Generate high-resolution soil moisture and soil temperature fields.
11. Produce multiple ensemble realizations when probabilistic output is required.
12. Validate the generated fields against the high-resolution reference dataset.
13. Produce spatial maps, difference fields, time-series diagnostics, and statistical verification results.

## Workflow Diagram

The overall workflow of the repository is illustrated below:

<p align="center">
  <img src="docs/IMG2.png" alt="Workflow Diagram" width="300">
</p>

The workflow can be summarized as:

**ERA5-Land conditioning data → MetCoOp domain preprocessing → high-resolution target preparation → normalization/Zarr generation → conditional diffusion-model training → stochastic high-resolution generation → validation and uncertainty analysis**

## Expected Output

The framework produces high-resolution soil moisture and soil temperature fields over the MetCoOp/Nordic domain.

Typical model outputs include:

- High-resolution soil moisture maps.
- High-resolution soil temperature maps.
- Ensemble-mean fields.
- Ensemble-spread fields.
- Reference-minus-generated difference maps.
- RMSE, MAE, and bias statistics.
- Seasonal and event-based evaluation plots.

## Applications

The generated high-resolution land-surface information can support applications including:

- Numerical weather prediction.
- Data assimilation.
- Hydrological modelling.
- Land-surface modelling.
- Regional climate analysis.
- Drought and wetness monitoring.
- Probabilistic prediction and uncertainty estimation.
- Development of AI-based high-resolution Earth-system modelling methods.

## Repository Structure

A typical repository organization can be represented as:

```text
Diffusion_Soil-Moisture/
├── config/                 # Model and experiment configuration files
├── data/                   # Input and processed datasets
├── docs/                   # Documentation and workflow figures
├── scripts/                # Preprocessing, training and evaluation scripts
├── src/                    # Diffusion-model source code
├── output/                 # Generated samples and evaluation products
├── README.md               # Repository documentation
└── requirements.txt        # Python dependencies
```

The exact directory structure may differ depending on the experiment and computing environment.

## Authors

- Swapan Mallick (SMHI, Sweden)
