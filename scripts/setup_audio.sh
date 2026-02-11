#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Zem Audio Module Setup ===${NC}"

# 1. Install System Dependencies (Linux)
if [ "$(uname)" == "Linux" ]; then
    echo -e "${YELLOW}Checking system dependencies...${NC}"
    # Minimal check for common missing libs
    if ! ldconfig -p | grep -q libsndfile; then
        echo "Installing libsndfile1..."
        sudo apt-get update && sudo apt-get install -y libsndfile1
    fi
fi

# 2. Install Python Dependencies (k2, icefall)
echo -e "${YELLOW}Installing Audio Libraries (k2, icefall)...${NC}"

# Function to get CUDA version for k2
get_cuda_version() {
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//'
    else
        echo "cpu"
    fi
}

# Determine install command based on availability
echo "Detected platform: $(uname -s)"
# Note: k2 pre-compiled wheels are tricky. 
# We try to install from k2-fsa verify compatible torch version or build from source.

# Installing k2
echo "Installing k2..."
# Try installing standard version compatible with torch
pip install k2==1.24.4+cuda11.8.0 -f https://k2-fsa.github.io/k2/cuda.html || \
pip install k2 

# Installing kaldifeat
echo "Installing kaldifeat..."
pip install kaldifeat

# Installing icefall
echo "Installing icefall..."
pip install git+https://github.com/k2-fsa/icefall.git

# Installing other deps
pip install lhotse sentencepiece

# 3. Download Models
MODEL_DIR="src/xfmr_zem/audio/components/asr/models"
mkdir -p "$MODEL_DIR"

echo -e "${YELLOW}Downloading Vietnamese ASR Model (Zipformer)...${NC}"
# Logic to download specific model
# Example: huggingface-cli download or wget
# For now, we put a placeholder or basic download if URL is known.
# Assuming model zzasdf/viet_iter3_pseudo_label

if [ ! -d "$MODEL_DIR/viet_iter3_pseudo_label" ]; then
    echo "Downloading model from HuggingFace..."
    pip install -U huggingface_hub
    huggingface-cli download zzasdf/viet_iter3_pseudo_label \
        --local-dir "$MODEL_DIR/viet_iter3_pseudo_label" \
        --local-dir-use-symlinks False
else
    echo "Model already exists at $MODEL_DIR/viet_iter3_pseudo_label"
fi

echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo "You can now run: zem server servers/audio"
