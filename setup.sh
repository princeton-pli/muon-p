#!/usr/bin/env bash
# Environment setup for the Muon training / evaluation code.
#
# Creates a Python virtual environment, installs the dependencies from
# requirements.txt, optionally builds FlashAttention-2, and runs an import
# smoke test to confirm everything loads.
#
# Usage:
#   ./setup.sh                  # create ./.venv and install everything
#   ENV_DIR=/path/to/venv ./setup.sh
#   SKIP_FLASH=1 ./setup.sh     # skip the (slow) flash-attn build
#   PYTHON=python3.11 ./setup.sh
#
# After it finishes, activate the environment with:
#   source .venv/bin/activate

set -euo pipefail

PYTHON="${PYTHON:-python3}"
ENV_DIR="${ENV_DIR:-.venv}"
SKIP_FLASH="${SKIP_FLASH:-0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

info() { echo "[setup] $*"; }

# ----------------------------------------------------------------------------
# 1. Create the virtual environment
# ----------------------------------------------------------------------------
if [ ! -d "$ENV_DIR" ]; then
  info "Creating virtual environment at $ENV_DIR"
  "$PYTHON" -m venv "$ENV_DIR"
else
  info "Reusing existing virtual environment at $ENV_DIR"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# ----------------------------------------------------------------------------
# 2. Install core dependencies
# ----------------------------------------------------------------------------
info "Upgrading pip / wheel / setuptools"
pip install --upgrade pip wheel setuptools

info "Installing requirements.txt"
pip install -r requirements.txt

# ----------------------------------------------------------------------------
# 3. FlashAttention-2 (needed by train_hf.py)
# ----------------------------------------------------------------------------
if [ "$SKIP_FLASH" = "1" ]; then
  info "Skipping flash-attn install (SKIP_FLASH=1). train_hf.py needs it; train_gsm8k.py does not."
else
  info "Installing flash-attn==${FLASH_ATTN_VERSION} (this can take several minutes)"
  pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation \
    || info "flash-attn build failed; install it manually later (see requirements.txt)."
fi

# ----------------------------------------------------------------------------
# 4. Smoke test: make sure all project modules import
# ----------------------------------------------------------------------------
info "Running import smoke test"
python - <<'PY'
import importlib

# Third-party deps.
for mod in ["torch", "transformers", "datasets", "accelerate", "numpy", "yaml"]:
    importlib.import_module(mod)

# Project modules.
for mod in [
    "utils",
    "all_arguments",
    "dual_optimizer",
    "perf_bench",
    "eval_math500",
    "eval_gsm8k",
    "train_hf",
    "train_gsm8k",
]:
    importlib.import_module(mod)

import torch
print(f"[setup] OK  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
PY

info "Done. Activate with:  source $ENV_DIR/bin/activate"
