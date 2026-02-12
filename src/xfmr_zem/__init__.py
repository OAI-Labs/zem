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



from xfmr_zem.client import PipelineClient

__all__ = [
    "PipelineClient",
]
