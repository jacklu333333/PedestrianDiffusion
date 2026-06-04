#!/bin/zsh
# Note: To keep the environment activated after the script finishes,
# run this script using "source":
#   source env_setup.sh

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error

# --- Mode Selection ---
echo -e "\033[36mSelect Environment Mode:\033[0m"
echo "  1) CUDA (Default, for systems with NVIDIA GPUs)"
echo "  2) CPU-Only (For Raspberry Pi, Mac, or systems without NVIDIA GPUs)"
echo -n "Enter choice [1 or 2]: "
read -r mode_choice

if [[ "$mode_choice" == "2" ]]; then
    USE_CUDA=false
    echo -e "\033[32m=> Mode set to: CPU-Only\033[0m"
else
    USE_CUDA=true
    echo -e "\033[32m=> Mode set to: CUDA\033[0m"
fi
echo

# --- Configuration ---
ENV_NAME="diffusion_202605post1"
PYTHON_VERSION="3.13"   # Latest stable release as of May 10, 2026
PY_SHORT="cp313"         # Updated for Python 3.13 wheel filenames

# --- Trap for Cleanup on Failure ---
SETUP_SUCCESS=false

cleanup_on_error() {
    if [ "$SETUP_SUCCESS" = false ]; then
        echo -e "\033[31m"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!!! ACCESS DENIED / SETUP FAILED: Cleaning up... !!!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo -e "\033[0m"
        
        # Deactivate if active (silence errors if not active)
        set +e
        conda deactivate 2>/dev/null
        
        # Remove the partial environment
        if conda info --envs | grep -q "^${ENV_NAME} "; then
            echo "Removing partial environment '$ENV_NAME'..."
            conda env remove -n "$ENV_NAME" -y
        fi
        set -e
        echo -e "\033[1;31mCleanup complete. Please fix the errors and try again.\033[0m"
    fi
}

# Trap EXIT signal (happens on successful exit OR error exit)
trap cleanup_on_error EXIT

# Version definitions
CUDA_VERSION_MAJOR="13"
CUDA_VERSION_MINOR="2"
CUDA_FULL_VERSION="${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}"
TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
TORCHAUDIO_VERSION="2.11.0"
CU_SUFFIX="cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}" # e.g. cu132

RAPIDS_VERSION="26.04" # Latest RAPIDS release for the current cycle

MAMBA_VERSION="2.3.2.post1" # Latest stable version
CAUSAL_CONV1D_VERSION="1.6.2.post1" # Current stable build


echo -e "\033[34m=== Starting Environment Setup for ${ENV_NAME} ===\033[0m"

# --- Check Conda ---
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH."
    exit 1
fi

# --- Create Environment ---
# Check if environment exists
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    # Depending on bash/zsh differences, -p and -n might behave oddly, but keeping original syntax
    read -p "Do you want to remove and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "$ENV_NAME"
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    else
        echo "Using existing environment. Proceeding with updates..."
    fi
else
    echo "Creating environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# --- Activate Environment ---
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

set +u
conda activate "$ENV_NAME"
set -u

echo "Activated environment: ${ENV_NAME}"

# Set deterministic cuBLAS workspace to ensure reproducible runs when using CUDA
if [ "$USE_CUDA" = true ]; then
    echo "Setting CUBLAS_WORKSPACE_CONFIG for environment '${ENV_NAME}'..."
    conda env config vars set -n "$ENV_NAME" CUBLAS_WORKSPACE_CONFIG=":4096:8"
    # Re-activate to apply the new env var to this shell session
    set +u
    conda deactivate >/dev/null 2>&1 || true
    conda activate "$ENV_NAME"
    set -u
fi

# --- Set Hardware-Specific Dependencies ---
if [ "$USE_CUDA" = true ]; then
    # ONNX_PKG="onnxruntime-gpu==1.24.2"
    ONNX_PKG=""
    
    # --- Install CUDA Toolkit ---
    echo -e "\033[34mInstalling CUDA Toolkit ${CUDA_FULL_VERSION}...\033[0m"
    set +u
    conda install -c nvidia -y \
    "cuda-toolkit=${CUDA_FULL_VERSION}.*" \
    "cuda-nvcc=${CUDA_FULL_VERSION}.*" \
    "cuda-version=${CUDA_FULL_VERSION}" \
    "cuda-compiler=${CUDA_FULL_VERSION}.*" \
    "cudnn"
    pip install "tensorrt==10.1.*"
    set -u
    
    # --- Install PyTorch and RAPIDS Together ---
    echo -e "\033[34mInstalling PyTorch and RAPIDS together to resolve CUDA dependencies...\033[0m"
    pip install \
    --extra-index-url "https://download.pytorch.org/whl/${CU_SUFFIX}" \
    --extra-index-url=https://pypi.nvidia.com \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "cudf-cu${CUDA_VERSION_MAJOR}==${RAPIDS_VERSION}.*" \
    "dask-cudf-cu${CUDA_VERSION_MAJOR}==${RAPIDS_VERSION}.*" \
    "cuml-cu${CUDA_VERSION_MAJOR}==${RAPIDS_VERSION}.*" \
    "cugraph-cu${CUDA_VERSION_MAJOR}==${RAPIDS_VERSION}.*" \
    "nx-cugraph-cu${CUDA_VERSION_MAJOR}==${RAPIDS_VERSION}.*" \
    "cuxfilter-cu${CUDA_VERSION_MAJOR}==${RAPIDS_VERSION}.*" \
    "cucim-cu${CUDA_VERSION_MAJOR}==${RAPIDS_VERSION}.*" \
    "pylibraft-cu${CUDA_VERSION_MAJOR}==${RAPIDS_VERSION}.*" \
    "raft-dask-cu${CUDA_VERSION_MAJOR}==${RAPIDS_VERSION}.*" \
    "cuvs-cu${CUDA_VERSION_MAJOR}==${RAPIDS_VERSION}.*"
    
    echo -e "\033[34mInstalling CUDA-specific libraries (bitsandbytes, pycuda)...\033[0m"
    pip install bitsandbytes pycuda
    pip install onnxruntime-gpu==1.24.2 --index-url https://visualstudio.com
    pip install tensorrt tensorrt-cu13 --extra-index-url https://pypi.nvidia.com
    # pip install tensorrt==10.0.1 tensorrt-cu12==10.0.1 --extra-index-url https://nvidia.com
    
else
    ONNX_PKG="onnxruntime==1.24.2"
    
    # --- Install PyTorch for CPU ---
    echo -e "\033[34mInstalling PyTorch (CPU version)...\033[0m"
    pip install \
    --extra-index-url "https://download.pytorch.org/whl/cpu" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"
    
    echo -e "\033[33mSkipping RAPIDS installation (cuDF, cuML, etc.) as they require an NVIDIA GPU.\033[0m"
fi

# --- Install LieGroups ---
echo -e "\033[34mInstalling liegroups...\033[0m"
pip install git+https://github.com/utiasSTARS/liegroups@master --use-pep517

# --- Install General Dependencies ---
echo "Installing general python dependencies..."
# pip install \
# "transformers==5.2.0" "diffusers==0.36.0" "accelerate==1.12.0" "datasets==4.6.0" "safetensors==0.7.0" \
# "timm==1.0.25" "einops==0.8.2" "peft==0.18.1" "lightning==2.6.1" "pytorch-lightning==2.6.1" "deepspeed==0.18.6" \
# "wandb==0.25.0" "tensorboard==2.20.0" "onnx==1.20.1" "${ONNX_PKG}" \
# "numpy==2.2.6" "pandas==2.3.3" "scipy==1.17.1" "scikit-learn==1.8.0" "scikit-image==0.25.2" \
# "matplotlib==3.10.8" "seaborn==0.13.2" "plotly==6.5.2" "bokeh==3.6.3" "altair==6.0.0" "streamlit==1.54.0" \
# "holoviews==1.20.2" "datashader==0.18.2" "panel==1.7.5" "hvplot==0.12.2" "pyviz_comms==3.0.6" \
# "geopandas==1.1.2" "shapely==2.0.7" "pyproj==3.7.2" "xarray==2026.2.0" \
# "ipykernel==7.2.0" "ipywidgets==8.1.8" "jupyterlab==4.5.5" "notebook==7.5.4" \
# "pydantic==2.12.5" "hydra-core==1.3.2" "omegaconf==2.3.0" "PyYAML==6.0.3" "h5py==3.15.1" \
# "tqdm==4.67.3" "click==8.3.1" "rich==14.3.3" "coloredlogs==15.0.1" "colorlog==6.10.1" \
# "requests==2.32.5" "httpx==0.28.1" "aiohttp==3.13.3" "Flask==3.1.3" "SQLAlchemy==2.0.47" "sqlmodel==0.0.37" \
# "AHRS==0.4.0" "numpy-quaternion==2024.0.13" "colored==2.3.1" "kornia==0.8.2" "overrides==7.7.0" "pyvista==0.47.1" "tensorboardX==2.6.4" "flatten-dict==0.4.2" kaleido black isort bitsandbytes scienceplots imageio-ffmpeg anywidget pypose rosbags zarr  onnxconverter-common ipympl pycuda litlogger
pip install \
transformers diffusers accelerate datasets safetensors \
timm einops peft lightning pytorch-lightning \
wandb tensorboard onnx "${ONNX_PKG}" \
numpy pandas scipy scikit-learn scikit-image \
matplotlib seaborn plotly bokeh altair streamlit \
holoviews datashader panel hvplot pyviz_comms \
geopandas shapely pyproj xarray \
ipykernel ipywidgets jupyterlab notebook \
pydantic hydra-core omegaconf PyYAML h5py \
tqdm click rich coloredlogs colorlog \
requests httpx aiohttp Flask SQLAlchemy sqlmodel \
AHRS numpy-quaternion colored kornia overrides pyvista tensorboardX flatten-dict \
kaleido black isort scienceplots imageio-ffmpeg anywidget pypose rosbags zarr onnxconverter-common ipympl litlogger \
onnx2tf tensorflow tf-keras onnx-graphsurgeon ai-edge-litert sng4onnx onnxsim onnxscript thop qonnx deepspeed

# Optional: Ensure choreo_get_chrome function exists in your env if keeping this line
choreo_get_chrome

# --- Install Mamba & Causal Conv1d ---
if [ "$USE_CUDA" = true ]; then
    echo "Detecting PyTorch ABI compatibility..."
    ABI_STATUS=$(python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)")
    echo "PyTorch _GLIBCXX_USE_CXX11_ABI Status: $ABI_STATUS"
    
    if [ "$ABI_STATUS" = "True" ]; then
        ABI_STR="cxx11abiTRUE"
    else
        ABI_STR="cxx11abiFALSE"
    fi
    
    # Detect Python version for wheel (e.g., cp311)
    PY_SHORT=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    echo "Detected Python Short Version: ${PY_SHORT}"
    
    echo -e "\033[34mInstalling Mamba SSM and Causal Conv1d...\033[0m"
    
    TORCH_VER_MM=$(echo $TORCH_VERSION | cut -d. -f1,2)
    CUDA_VER_MM="cu${CUDA_VERSION_MAJOR}"
    MAMBA_CUDA_TAG="cu${CUDA_VERSION_MAJOR}"
    
    MAMBA_WHEEL_NAME="mamba_ssm-${MAMBA_VERSION}+${MAMBA_CUDA_TAG}torch${TORCH_VER_MM}${ABI_STR}-${PY_SHORT}-${PY_SHORT}-linux_x86_64.whl"
    CAUSAL_WHEEL_NAME="causal_conv1d-${CAUSAL_CONV1D_VERSION}+${MAMBA_CUDA_TAG}torch${TORCH_VER_MM}${ABI_STR}-${PY_SHORT}-${PY_SHORT}-linux_x86_64.whl"
    
    MAMBA_URL="https://github.com/state-spaces/mamba/releases/download/v${MAMBA_VERSION}/${MAMBA_WHEEL_NAME}"
    CAUSAL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_VERSION}/${CAUSAL_WHEEL_NAME}"
    
    echo "Attempting to install from wheels..."
    echo "  Target Mamba URL: $MAMBA_URL"
    echo "  Target Causal URL: $CAUSAL_URL"
    
    # Install Causal Conv1d
    if pip install "$CAUSAL_URL"; then
        echo "Successfully installed causal_conv1d from wheel."
    else
        echo "Wheel installation failed for causal_conv1d."
        echo "Falling back to building from source (may take a few minutes)..."
        if pip install "git+https://github.com/Dao-AILab/causal-conv1d.git@v${CAUSAL_CONV1D_VERSION}" --no-build-isolation; then
            echo "Successfully installed causal_conv1d from source."
        else
            echo "ERROR: Failed to install causal_conv1d from either wheel or source."
            exit 1
        fi
    fi
    
    # Install Mamba SSM
    if pip install "$MAMBA_URL"; then
        echo "Successfully installed mamba_ssm from wheel."
    else
        echo "Wheel installation failed for mamba_ssm."
        echo "Falling back to building from source (may take a few minutes)..."
        if pip install "git+https://github.com/state-spaces/mamba.git@v${MAMBA_VERSION}" --no-build-isolation; then
            echo "Successfully installed mamba_ssm from source."
        else
            echo "ERROR: Failed to install mamba_ssm from either wheel or source."
            exit 1
        fi
    fi
else
    echo -e "\033[33mSkipping Mamba SSM and Causal Conv1d installation.\033[0m"
    echo "These packages require CUDA for their custom kernels and are currently not supported out-of-the-box on CPU-only ARM64 environments like the Raspberry Pi."
fi

# --- Success Marker ---
SETUP_SUCCESS=true

echo "=== Setup Completed Successfully ==="

# --- Enter Environment (if not sourced) ---
if [[ -z "${ZSH_EVAL_CONTEXT:-}" ]]; then
    echo "Launching new shell with environment activated..."
    TMP_ZDOTDIR=$(mktemp -d)
    
    cat <<EOF > "${TMP_ZDOTDIR}/.zshrc"
if [ -f "\${HOME}/.zshrc" ]; then
    source "\${HOME}/.zshrc"
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
trap "rm -rf ${TMP_ZDOTDIR}" EXIT
EOF
    
    export ZDOTDIR="${TMP_ZDOTDIR}"
    exec zsh
else
    echo "Environment '$ENV_NAME' finished installing."
fi
