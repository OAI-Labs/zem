# Changelog

All notable changes to this project will be documented in this file.

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
