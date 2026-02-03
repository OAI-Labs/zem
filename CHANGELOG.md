# Changelog

All notable changes to this project will be documented in this file.

## [0.2.9] - 2026-02-03

### Added
- **Voice Processing Module**: Added a new `voice` server with Automatic Speech Recognition (ASR) support using OpenAI Whisper.
- **Voice Transcription Tool**: Introduced `transcribe` tool for high-quality audio-to-text conversion.

## [0.2.8] - 2026-02-03

### Fixed
- **Parameter Support**: Fixed dot-notation support for hierarchical parameters in pipeline configurations (e.g., `ocr.temp_dir`).
- **OCR Server**: Added detailed debug logging for temporary file operations.

## [0.2.7] - 2026-02-03

### Added
- **Configurable OCR Parameters**: Added `scanned_threshold`, `zoom`, and `temp_dir` parameters to the OCR server for finer control over PDF processing.

## [0.2.6] - 2026-02-03

### Added
- **PDF Extraction & Scanning**: Added support for processing multi-page PDFs in the OCR server. It automatically extracts digital text and falls back to OCR for scanned pages using `PyMuPDF`.

## [0.2.5] - 2026-02-03

### Added
- **Vietnamese OCR**: Integrated specialized Deep-ocr DocumentPipeline for high-accuracy Vietnamese text extraction with layout analysis.
- **Improved OCR Dependencies**: Added `pdfplumber`, `ruamel.yaml`, and `cachetools` to resolve OCR server tool errors.

## [0.2.4] - 2026-02-02

### Fixed
- **CI Permissions**: Explicitly granted write permissions in workflow to bypass repository UI restrictions.

## [0.2.3] - 2026-02-02

### Added
- **GitHub Release Integration**: automated creation of formal GitHub Releases with binary assets.

## [0.2.2] - 2026-02-02

### Added
- **CI/CD Automation**: Integrated GitHub Actions for automated PyPI publishing on tag creation.
- **Improved Metadata**: Linked PyPI package to GitHub repository, issues, and changelog.

## [0.2.1] - 2026-02-02

### Added
- **Parallel Orchestration**: New `ParallelLocalOrchestrator` allowing true concurrent execution of independent DAG branches.
- **Dynamic Project Bootstrapping**: `zem init` command to create standalone project structures with sample agents.
- **Adaptive Caching**: Step-level cache control (`cache: true/false`) in pipeline YAML.
- **Cloud & Parquet Support**: Native loading from S3, GCS, and HTTP; high-performance Parquet IO.
- **Frontier LLM Integration**: `llm` server with Ollama and OpenAI support for smart curation (PII masking, classification).
- **Data Sinks**: Seamless export to Hugging Face Hub and Vector Databases (Pinecone, Milvus).
- **Enhanced Observability**: 
  - `zem preview --diff` to compare artifact versions.
  - `zem preview --sample` for random data sampling.
  - `zem dashboard` shortcut for ZenML visualization.
- **Improved Validation**: Pydantic v2 schemas for robust pipeline configuration.

### Fixed
- Fixed broken flavor icons in ZenML Dashboard.
- Improved server path resolution to handle both internal and project-local servers.
- Fixed f-string escaping and path joining in project bootstrap logic.
- Standardized tool registration to avoid "multiple values for argument 'name'" errors.

## [0.1.0] - 2026-01-26

### Added
- Initial foundation with ZenML + MCP integration.
- Bridge servers for NeMo Curator and DataJuicer.
- Basic CLI with `run`, `list-tools`, and `preview`.
- Support for sequential pipelines.
