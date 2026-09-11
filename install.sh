#!/bin/bash
set -e

# Always resolve paths relative to this script, regardless of where it's called from
cd "$(dirname "$0")"

echo ""
echo "========================================"
echo "   SegmentME Installer (Linux / macOS)"
echo "========================================"
echo ""

# Check Python 3.10+
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10 or newer and try again."
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required (found $PY_VER)."
    exit 1
fi

echo "Python $PY_VER detected."
echo ""

# Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# PyTorch CPU-only (macOS: the default wheel, which also covers Apple Metal)
echo "[2/5] Installing PyTorch (CPU)..."
if [ "$(uname)" = "Darwin" ]; then
    pip install --quiet --no-cache-dir torch torchvision
else
    pip install --quiet --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
fi

# Main requirements
echo "[3/5] Installing dependencies..."
pip install --quiet --no-cache-dir -r requirements.txt

# SAM2, SAM3 and DEXTR (not on PyPI; installed from GitHub archives, no git needed)
echo "[4/5] Installing SAM2, SAM3 and DEXTR..."
SAM2_BUILD_CUDA=0 pip install --quiet --no-cache-dir \
    "https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip"
pip install --quiet --no-cache-dir \
    "https://github.com/facebookresearch/sam3/archive/refs/heads/main.zip" einops pycocotools
DEXTR_TMP=$(mktemp -d)
curl -sL "https://github.com/StevetheGreek97/DEXTR-SMe/archive/refs/heads/master.zip" -o "$DEXTR_TMP/dextr.zip"
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$DEXTR_TMP/dextr.zip" "$DEXTR_TMP"
touch "$DEXTR_TMP"/DEXTR-SMe-*/src/DEXTR/__init__.py
pip install --quiet --no-cache-dir "$DEXTR_TMP"/DEXTR-SMe-*/
rm -rf "$DEXTR_TMP"

# Model checkpoints live in models/ next to the app (see services/model_store.py).
# Only SAM2 Tiny is fetched here; everything else can be downloaded from
# Settings -> Models inside the app.
echo "[5/5] Downloading the default SAM2 Tiny model..."
mkdir -p models
if [ ! -f "models/sam2_hiera_tiny.pt" ]; then
    echo "  Downloading sam2_hiera_tiny.pt (~155 MB)..."
    curl -L --progress-bar \
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt" \
        -o "models/sam2_hiera_tiny.pt"
else
    echo "  sam2_hiera_tiny.pt already present, skipping download."
fi

# SAM3 checkpoint (gated on Hugging Face -- needs an account with access)
if [ ! -f "models/sam3.pt" ]; then
    echo "  Trying to download sam3.pt (~3.4 GB, gated)..."
    if .venv/bin/hf download facebook/sam3 sam3.pt --local-dir models 2>/dev/null; then
        echo "  sam3.pt downloaded."
    else
        echo "  SAM3 not downloaded (needs Hugging Face access). To add it later:"
        echo "    request access at https://huggingface.co/facebook/sam3, then run"
        echo "    .venv/bin/hf auth login"
        echo "    .venv/bin/hf download facebook/sam3 sam3.pt --local-dir models"
        echo "    or import the file from Settings -> Models inside the app."
    fi
else
    echo "  sam3.pt already present, skipping download."
fi

echo ""
echo "========================================"
echo "   Installation complete!"
echo "   Run ./run.sh to start SegmentME."
echo "   Other models (SAM2 Small/Large, SAM2.1, DEXTR):"
echo "   Settings -> Models inside the app."
echo "========================================"
echo ""
