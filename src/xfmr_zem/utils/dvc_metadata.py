"""
DVC Metadata Extractor & ZenML Integration
==========================================

This module provides utilities for extracting DVC metadata and linking it
with ZenML experiment tracking for full reproducibility.

Features:
- Extract MD5 hash from DVC tracked files
- Link DVC data versions with ZenML pipeline runs
- Create comprehensive lineage metadata
- Support for both local and remote DVC storage

Usage:
    from xfmr_zem.utils.dvc_metadata import DVCMetadataExtractor, DVCZenMLBridge
    
    # Get DVC hash for a file
    hash = DVCMetadataExtractor.get_dvc_hash("data/dataset.parquet")
    
    # Create lineage metadata
    lineage = DVCMetadataExtractor.create_lineage_metadata(
        data_path="data/dataset.parquet",
        artifact_id="abc-123",
        run_id="run-456"
    )
    
    # Log to ZenML
    bridge = DVCZenMLBridge()
    bridge.log_dvc_metadata_to_run(data_path, run_id)
"""

import hashlib
import os
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger


class DVCMetadataExtractor:
    """
    Extract and manage DVC metadata for data versioning.
    
    This class provides static methods to:
    - Extract MD5 hashes from .dvc files
    - Compute hashes for non-tracked files
    - Get git commit information
    - Create comprehensive lineage metadata
    """
    
    # Max size (bytes) for manual hash computation. Files/dirs larger than this
    # will return None instead of blocking the pipeline. Default: 2 GB.
    MAX_HASH_SIZE: int = 2 * 1024 * 1024 * 1024

    @staticmethod
    def get_dvc_hash(file_path: str) -> Optional[str]:
        """
        Extract MD5 hash from .dvc file or compute for raw file.
        
        Priority: .dvc file hash > manual computation (with size limit).
        Files/dirs exceeding MAX_HASH_SIZE without a .dvc file will return None.
        
        Args:
            file_path: Path to the data file (not the .dvc file)
            
        Returns:
            MD5 hash string or None if file doesn't exist or exceeds size limit
            
        Example:
            >>> DVCMetadataExtractor.get_dvc_hash("data/train.parquet")
            '6ef00d6c75703b1f7b566224c5dc5542'
        """
        dvc_file = Path(str(file_path) + ".dvc")
        
        # Try to read from .dvc file first (fast path - no I/O on data)
        if dvc_file.exists():
            try:
                with open(dvc_file, "r") as f:
                    dvc_meta = yaml.safe_load(f)
                    outs = dvc_meta.get("outs", [])
                    if outs and len(outs) > 0:
                        md5 = outs[0].get("md5")
                        if md5:
                            logger.debug(f"DVC hash found for {file_path}: {md5}")
                            return md5
            except Exception as e:
                logger.warning(f"Error reading .dvc file {dvc_file}: {e}")
        
        # Fallback: compute hash manually (with size guard)
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            if file_path_obj.is_dir():
                return DVCMetadataExtractor._compute_dir_hash(file_path)
            # Single file size guard
            file_size = file_path_obj.stat().st_size
            if file_size > DVCMetadataExtractor.MAX_HASH_SIZE:
                logger.warning(
                    f"File {file_path} ({file_size / 1e9:.1f} GB) exceeds MAX_HASH_SIZE. "
                    f"Track with 'zem data add' first for fast hash lookup."
                )
                return None
            return DVCMetadataExtractor._compute_md5(file_path)
        
        logger.warning(f"File not found: {file_path}")
        return None
    
    @staticmethod
    def _compute_md5(file_path: str, chunk_size: int = 8192) -> str:
        """
        Compute MD5 hash for a single file.
        
        Args:
            file_path: Path to the file
            chunk_size: Size of chunks to read (default 8KB)
            
        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def _compute_dir_hash(dir_path: str) -> Optional[str]:
        """
        Compute combined hash for a directory.
        
        Uses file names + sizes as a lightweight fingerprint instead of reading
        every byte. For exact content hashes, track the directory with DVC first
        (``zem data add <dir>``), which stores the authoritative hash in the
        ``.dvc`` file and is read by :meth:`get_dvc_hash` without any I/O on
        the data itself.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            Combined MD5 hash string (suffixed with ``.dir``), or None if the
            directory exceeds MAX_HASH_SIZE.
        """
        dir_path_obj = Path(dir_path)

        # Compute total size first to enforce the size guard
        total_size = sum(f.stat().st_size for f in dir_path_obj.rglob("*") if f.is_file())
        if total_size > DVCMetadataExtractor.MAX_HASH_SIZE:
            logger.warning(
                f"Directory {dir_path} ({total_size / 1e9:.1f} GB) exceeds MAX_HASH_SIZE. "
                f"Track with 'zem data add' first for fast hash lookup."
            )
            return None

        hash_md5 = hashlib.md5()

        # Sort files for consistent hashing
        files = sorted(dir_path_obj.rglob("*"))
        for file_path in files:
            if file_path.is_file():
                rel_path = file_path.relative_to(dir_path_obj)
                # Include relative path + size for structure/content awareness
                hash_md5.update(str(rel_path).encode())
                hash_md5.update(str(file_path.stat().st_size).encode())
                hash_md5.update(str(file_path.stat().st_mtime_ns).encode())
        
        return hash_md5.hexdigest() + ".dir"
    
    @staticmethod
    def get_git_commit() -> Optional[str]:
        """
        Get current git commit hash.
        
        Returns:
            Git commit hash (40 chars) or None if not in a git repo
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not get git commit: {e}")
        return None
    
    @staticmethod
    def get_git_branch() -> Optional[str]:
        """
        Get current git branch name.
        
        Returns:
            Branch name or None if not in a git repo
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    @staticmethod
    def get_dvc_remote_info() -> Optional[Dict[str, str]]:
        """
        Get DVC remote configuration.
        
        Returns:
            Dict with remote name and URL, or None if not configured
        """
        try:
            result = subprocess.run(
                ["dvc", "remote", "list"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                remotes = {}
                for line in lines:
                    # DVC remote list can use tabs or spaces as separator
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        remotes[parts[0].strip()] = parts[1].strip()
                    else:
                        # Fallback: split on whitespace
                        parts = line.split()
                        if len(parts) >= 2:
                            remotes[parts[0].strip()] = parts[1].strip()
                return remotes if remotes else None
        except Exception as e:
            logger.debug(f"Could not get DVC remote info: {e}")
        return None
    
    @staticmethod
    def get_file_stats(file_path: str) -> Dict[str, Any]:
        """
        Get file statistics.
        
        Args:
            file_path: Path to the file or directory
            
        Returns:
            Dict with size, file count, and modification time
        """
        path = Path(file_path)
        stats = {
            "path": str(path.absolute()),
            "exists": path.exists(),
        }
        
        if not path.exists():
            return stats
        
        if path.is_file():
            stat = path.stat()
            stats.update({
                "type": "file",
                "size_bytes": stat.st_size,
                "size_human": DVCMetadataExtractor._human_readable_size(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        elif path.is_dir():
            total_size = 0
            file_count = 0
            for f in path.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
                    file_count += 1
            stats.update({
                "type": "directory",
                "size_bytes": total_size,
                "size_human": DVCMetadataExtractor._human_readable_size(total_size),
                "file_count": file_count,
            })
        
        return stats
    
    @staticmethod
    def _human_readable_size(size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"
    
    @staticmethod
    def create_lineage_metadata(
        data_path: str,
        artifact_id: str = None,
        run_id: str = None,
        step_name: str = None,
        extra_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive lineage metadata linking DVC and experiment tracking.
        
        Args:
            data_path: Path to the data file/directory
            artifact_id: ZenML artifact ID (optional)
            run_id: ZenML run ID (optional)
            step_name: Pipeline step name (optional)
            extra_metadata: Additional metadata to include
            
        Returns:
            Dict containing all lineage information
            
        Example:
            >>> lineage = DVCMetadataExtractor.create_lineage_metadata(
            ...     data_path="data/train.parquet",
            ...     run_id="abc-123",
            ...     step_name="data_loader"
            ... )
            >>> print(lineage["dvc_hash"])
            '6ef00d6c75703b1f7b566224c5dc5542'
        """
        file_stats = DVCMetadataExtractor.get_file_stats(data_path)
        
        metadata = {
            # DVC Information
            "dvc": {
                "hash": DVCMetadataExtractor.get_dvc_hash(data_path),
                "remote": DVCMetadataExtractor.get_dvc_remote_info(),
                "tracked": Path(str(data_path) + ".dvc").exists(),
            },
            
            # Git Information
            "git": {
                "commit": DVCMetadataExtractor.get_git_commit(),
                "branch": DVCMetadataExtractor.get_git_branch(),
            },
            
            # File Information
            "data": file_stats,
            
            # Experiment Tracking
            "experiment": {
                "artifact_id": artifact_id,
                "run_id": run_id,
                "step_name": step_name,
            },
            
            # Timestamp
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        }
        
        if extra_metadata:
            metadata["extra"] = extra_metadata
        
        return metadata
    
    @staticmethod
    def parse_dvc_file(dvc_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse a .dvc file and extract all metadata.
        
        Args:
            dvc_file_path: Path to the .dvc file
            
        Returns:
            Parsed DVC metadata or None if file doesn't exist
        """
        path = Path(dvc_file_path)
        if not path.exists():
            return None
        
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error parsing DVC file {dvc_file_path}: {e}")
            return None


class DVCZenMLBridge:
    """
    Bridge between DVC and ZenML for integrated experiment tracking.
    
    This class provides methods to:
    - Log DVC metadata to ZenML runs
    - Create artifacts with DVC lineage
    - Query experiments by data version
    """
    
    def __init__(self):
        """Initialize the bridge."""
        self._zenml_client = None
    
    @property
    def zenml_client(self):
        """Lazy load ZenML client."""
        if self._zenml_client is None:
            try:
                from zenml.client import Client
                self._zenml_client = Client()
            except ImportError:
                logger.error("ZenML not installed. Install with: uv add zenml")
                raise
        return self._zenml_client
    
    def log_dvc_metadata_to_run(
        self,
        data_path: str,
        run_id: str = None,
        artifact_name: str = None
    ) -> Dict[str, Any]:
        """
        Log DVC metadata to a ZenML run.
        
        Args:
            data_path: Path to the data file/directory
            run_id: ZenML run ID (uses current run if None)
            artifact_name: Name for the artifact metadata
            
        Returns:
            The logged metadata
        """
        from zenml import log_artifact_metadata
        
        lineage = DVCMetadataExtractor.create_lineage_metadata(
            data_path=data_path,
            run_id=run_id
        )
        
        artifact_name = artifact_name or f"dvc_lineage_{Path(data_path).stem}"
        
        try:
            log_artifact_metadata(
                artifact_name=artifact_name,
                metadata={
                    "dvc_hash": lineage["dvc"]["hash"],
                    "dvc_tracked": lineage["dvc"]["tracked"],
                    "git_commit": lineage["git"]["commit"],
                    "data_size": lineage["data"].get("size_human"),
                    "timestamp": lineage["timestamp"],
                }
            )
            logger.info(f"Logged DVC metadata for {data_path} to ZenML")
        except Exception as e:
            logger.warning(f"Could not log to ZenML: {e}")
        
        return lineage
    
    def get_runs_by_data_hash(self, dvc_hash: str) -> List[Dict[str, Any]]:
        """
        Find all ZenML runs that used a specific data version.
        
        Args:
            dvc_hash: DVC hash to search for
            
        Returns:
            List of run information dicts
        """
        # Note: This requires ZenML metadata search capability
        # Implementation depends on ZenML version and backend
        logger.warning("get_runs_by_data_hash requires ZenML Pro or custom metadata store")
        return []
    
    def create_reproducibility_script(
        self,
        run_id: str,
        output_path: str = "reproduce.sh"
    ) -> str:
        """
        Generate a script to reproduce a specific experiment run.
        
        Args:
            run_id: ZenML run ID to reproduce
            output_path: Path to save the script
            
        Returns:
            Path to the generated script
        """
        try:
            run = self.zenml_client.get_pipeline_run(run_id)
            
            # Get git commit from run metadata if available
            git_commit = None
            # Note: Actual implementation depends on how metadata was logged
            
            script_content = f"""#!/bin/bash
# Reproducibility script for ZenML run: {run_id}
# Generated: {datetime.now().isoformat()}

set -e

echo "🔄 Reproducing experiment run: {run_id}"

# 1. Checkout correct code version
{f'git checkout {git_commit}' if git_commit else '# Git commit not recorded'}

# 2. Pull correct data version
dvc pull

# 3. Run the pipeline
# uv run zem run <your_pipeline.yaml>

echo "✅ Environment ready for reproduction"
"""
            
            with open(output_path, "w") as f:
                f.write(script_content)
            
            os.chmod(output_path, 0o755)
            logger.info(f"Created reproducibility script: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating reproducibility script: {e}")
            raise


def log_data_version(
    data_path: str,
    step_name: str = None,
    extra_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Convenience function to log data version in a ZenML step.
    
    This is the recommended way to track data versions in Zem pipelines.
    
    Args:
        data_path: Path to the data file/directory
        step_name: Name of the current step
        extra_metadata: Additional metadata to log
        
    Returns:
        Lineage metadata dict
        
    Example:
        @step
        def my_data_step(data_path: str) -> Any:
            # Log the data version
            lineage = log_data_version(data_path, step_name="my_data_step")
            
            # Process data...
            return processed_data
    """
    return DVCMetadataExtractor.create_lineage_metadata(
        data_path=data_path,
        step_name=step_name,
        extra_metadata=extra_metadata
    )
