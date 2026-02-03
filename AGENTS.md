# Zem Project Architecture

## Overview
`Zem` is a unified data pipeline framework that leverages the **Model Context Protocol (MCP)** architecture to orchestrate data processing tasks across multiple domains. It integrates with **ZenML** for pipeline visualization and tracking.

## Core Components

### 1. MCP Servers (`src/xfmr_zem/server.py`)
- **Base Class**: `ZemServer` extends `FastMCP`.
- **Role**: Encapsulates specific domain logic (e.g., NeMo Curator, DataJuicer) as callable tools.
- **Constraints**: 
    - Must disable the startup banner (`show_banner=False`) to ensure clean JSON-RPC communication over stdio.
    - Tools should return dictionary results or standard MCP content structures.

### 2. Pipeline Client (`src/xfmr_zem/client.py`)
- **Role**: Acts as the orchestrator.
- **Functionality**:
    - Reads YAML configurations (`pipeline.yaml`, `server.yaml`).
    - Dynamically constructs a ZenML pipeline based on the configuration steps.
    - Manages the lifecycle of MCP server subprocesses.

### 3. ZenML Wrapper (`src/xfmr_zem/zenml_wrapper.py`)
- **Role**: Bridges ZenML steps with MCP tool execution.
- **Functionality**:
    - `mcp_generic_step`: A generic ZenML step that executes a specific tool on a target MCP server.
    - Implements robust manual subprocess communication for JSON-RPC over stdio (bypassing `mcp` library async issues).

## Configuration (`*.yaml`)
- **Pipeline Config**: Defines the sequence of steps (`server.tool`).
- **Server Config**: Maps logical server names to their implementation paths (Python scripts).

## Implemented Servers

- **NeMo Curator Server** (`src/xfmr_zem/servers/nemo_curator/server.py`):
    - Tools: `pii_removal`, `normalize`, `quality_filter`.
- **DataJuicer Server** (`src/xfmr_zem/servers/data_juicer/server.py`):
    - Tools: `clean_content`, `refining_filter`, `clean_html`, `clean_links`, `fix_unicode`, `whitespace_normalization`, `text_length_filter`, `language_filter`, `document_simhash_dedup`.
- **Profiler Server** (`src/xfmr_zem/servers/profiler/server.py`):
    - Tools: `profile_data` (Generates stats: word counts, null ratios, uniqueness).
- **LLM-Curation Server** (`src/xfmr_zem/servers/llm/server.py`):
    - Tools: `mask_pii`, `classify_domain`.
    - Providers: **Ollama** (local default), **OpenAI**.
- **Sinks Server** (`src/xfmr_zem/servers/sinks/server.py`):
    - Tools: `to_huggingface`, `to_vector_db`.
### 5. Unstructured Server (`servers/unstructured`)
Advanced document parsing for multimodal data ingestion.
- `parse_document`: Convert PDF, DOCX, HTML to structured text.
- `extract_tables`: Specifically isolate and extract table data from documents.

### 6. OCR Server (`servers/ocr`)
Unified OCR processing with multiple engine support (SOLID Strategy Pattern).
- `extract_text`: Extract text from images using different engines:
    - `tesseract`: Lightweight and fast.
    - `paddle`: Medium weight, high accuracy.
    - `qwen`: Heavy Vision-Language Model (Qwen3-VL-8B) for state-of-the-art OCR.
    - `viet`: Specialized Vietnamese OCR using built-in `deepdoc_vietocr` pipeline. Optimized for Vietnamese diacritics and document layout reconstruction.

## Orchestration & Concurrency

### 1. Sequential (Default)
Uses the standard ZenML local orchestrator. Steps are executed one-by-one.

### 2. Parallel Local (Optimized)
Uses the custom `ParallelLocalOrchestrator` (`src/xfmr_zem/orchestrators/parallel_local.py`).
- **Capability**: Executes independent DAG branches (trees) concurrently using multi-threading.
- **Setup**: `uv run zenml stack set parallel_stack`.

## Environment & Constraints

### 🔴 Port Usage (CRITICAL)
- **Allowed Port Range**: **8871 - 8879** (Environment restrictions)
- **ZenML Dashboard**: `uv run zenml up --port 8871`

### 🟢 Python Environment
- Use `uv run` to execute scripts in the correct environment.
- **PYTHONPATH**: When using the custom parallel orchestrator, you MUST include `src` in your path: `export PYTHONPATH=$PYTHONPATH:$(pwd)/src`.
