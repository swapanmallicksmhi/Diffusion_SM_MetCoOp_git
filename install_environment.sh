#!/bin/bash

set -e

ENV_NAME="climate_env"
PYTHON_VERSION="3.11"

echo "==============================================="
echo " Environment Installation"
echo "==============================================="

# ------------------------------------------------
# Check if mamba is available
# ------------------------------------------------
if ! command -v mamba &> /dev/null; then
    echo "Mamba not found. Please install Mambaforge first:"
    echo "https://github.com/conda-forge/miniforge#mambaforge"
    exit 1
fi

echo "Mamba found"

# ------------------------------------------------
# Create environment if it does not exist
# ------------------------------------------------
if ! mamba env list | grep -q "^${ENV_NAME} "; then
    echo "Creating environment: ${ENV_NAME}"
    mamba create -y -n ${ENV_NAME} python=${PYTHON_VERSION}
else
    echo "Environment '${ENV_NAME}' already exists "
fi

# ------------------------------------------------
# Activate environment
# ------------------------------------------------
echo "Activating environment: ${ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
mamba activate ${ENV_NAME}

# ------------------------------------------------
# Install core scientific stack
# ------------------------------------------------
echo "Installing scientific libraries..."
mamba install -y -c conda-forge \
    xarray \
    netcdf4 \
    h5netcdf \
    scipy \
    numpy \
    pandas \
    matplotlib \
    cartopy \
    proj \
    pyproj \
    shapely \
    geos \
    cfgrib \
    eccodes \
    dask \
    zarr

# ------------------------------------------------
# Install pip-based tools
# ------------------------------------------------
echo "Installing pip packages..."
pip install --upgrade pip
pip install cdsapi

# ------------------------------------------------
# Verify installation
# ------------------------------------------------
echo "Verifying installation..."
python - <<EOF
import xarray, cartopy, cdsapi
print("✔ xarray version:", xarray.__version__)
print("✔ cartopy version:", cartopy.__version__)
print("✔ cdsapi available")
EOF

echo "==============================================="
echo " Installation completed successfully"
echo " Activate with:  mamba activate ${ENV_NAME}"
echo "==============================================="
