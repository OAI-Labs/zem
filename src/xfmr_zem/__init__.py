"""
xfmr-zem (Zem)
==============

A unified data pipeline framework combining:
- Model Context Protocol (MCP): For modular, specialized processing servers.
- ZenML: For production-grade orchestration and pipeline tracking.

xfmr-zem allows you to build complex data processing workflows by connecting
multiple MCP servers as pipeline steps, all orchestrated by ZenML.

Key Features:
    * **Config-Driven Architecture**: Define your entire pipeline in a simple YAML configuration.
    * **MCP Server Integration**: Leverage any MCP-compatible server as a processing block.
    * **ZenML Orchestration**: Production-grade tracking, caching, and visualization of data flows.
    * **Multi-Domain Ready**: Designed for modular tasks like curation, extraction, and filtering.

Example:
    from xfmr_zem import PipelineClient

    # Initialize client with a pipeline configuration
    client = PipelineClient("configs/medical_pipeline.yaml")
    
    # Build and execute the ZenML pipeline
    client.run()
"""

__version__ = "0.1.0"
__author__ = "Khai Hoang"

import sys
from pathlib import Path

# Inject vendored diarizen_lib into sys.path
# This must be done here because running -m xfmr_zem.cli triggers this __init__ first,
# and subsequent imports (like zenml) might load pyannote before cli.py runs.
diarizen_lib = Path(__file__).parent / "audio" / "components" / "diarization" / "diarizen_lib"
if str(diarizen_lib) not in sys.path:
    start_index = 0
    if sys.path[0] == "": # local dir
         start_index = 1
    sys.path.insert(start_index, str(diarizen_lib))
    
    # Check if pyannote is already loaded and force unload if necessary
    # This prevents using the installed version if it was pre-loaded by dependencies
    if "pyannote" in sys.modules:
        to_unload = [m for m in sys.modules if m.startswith("pyannote")]
        for m in to_unload:
            del sys.modules[m]
    
    try:
        import pyannote
        # Manual namespace package merging if pkg_resources failed to find site-packages version
        if len(pyannote.__path__) == 1:
            import os
            for p in sys.path:
                if p == str(diarizen_lib):
                    continue
                potential_path = os.path.join(p, "pyannote")
                if os.path.isdir(potential_path) and potential_path not in pyannote.__path__:
                     pyannote.__path__.append(potential_path)
    except ImportError:
        pass

from xfmr_zem.client import PipelineClient

__all__ = [
    "PipelineClient",
]
