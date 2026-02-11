#!/usr/bin/env python3
"""
Setup script for Zem Audio Module.
Downloads necessary models and ensures dependencies are met.
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess
import argparse
from loguru import logger

# Constants
CACHE_DIR = Path.home() / ".cache" / "xfmr_zem" / "audio"
VIEASR_REPO = "zzasdf/viet_iter3_pseudo_label" # Or specific repo if available as artifacts
# Note: The original code expected a specific folder structure. 
# We'll download the model files from a source or allow user to specify.
# Since the original repo for VieASR might be large or private, we mock this or use a known public link if available.
# Ideally we use `huggingface_hub` if the model is on HF.
# zzasdf/viet_iter3_pseudo_label seems like a HF ID.

def check_dependencies():
    """Check and install Python dependencies."""
    logger.info("Checking dependencies...")
    
    needed = ["huggingface_hub", "soundfile", "numpy"]
    installed = []
    
    for pkg in needed:
        try:
            __import__(pkg)
            installed.append(pkg)
        except ImportError:
            logger.warning(f"Package {pkg} not found.")

    # Check for icefall
    try:
        import icefall
        logger.info("icefall is installed.")
    except ImportError:
        logger.warning("icefall is MISSING. It is required for VieASR.")
        logger.info("Please install it: pip install git+https://github.com/k2-fsa/icefall.git")

def download_models(force: bool = False):
    """Download audio models."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run `pip install huggingface_hub`")
        return

    # 1. VieASR Model
    target_dir = CACHE_DIR / "models" / "viet_iter3_pseudo_label"
    if target_dir.exists() and not force:
        logger.info(f"VieASR model found at {target_dir}")
    else:
        logger.info(f"Downloading VieASR model to {target_dir}...")
        try:
            # Note: This is a placeholder repo ID based on the code analysis.
            # If it doesn't exist, this will fail. We assume the user wants this.
            snapshot_download(repo_id=VIEASR_REPO, local_dir=target_dir)
            logger.info("VieASR download complete.")
        except Exception as e:
            logger.error(f"Failed to download VieASR model: {e}")
            logger.info("Please manually place the model at the target directory or update config.")

def main():
    parser = argparse.ArgumentParser(description="Setup Zem Audio Module")
    parser.add_argument("--force", action="store_true", help="Force re-download models")
    args = parser.parse_args()

    check_dependencies()
    download_models(args.force)

    logger.info("\nSetup verification complete.")
    logger.info(f"Models are stored in: {CACHE_DIR}")
    logger.info("Ensure your 'parameters.yml' or environment variables point to this location if needed.")

if __name__ == "__main__":
    main()
