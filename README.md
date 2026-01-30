# Zem

**Unified Data Pipeline Framework** combining **ZenML**, **NeMo Curator**, and **DataJuicer** for scalable, config-driven multi-domain data processing.

[🌐 Landing Page & Docs](https://khaihoang.github.io/xfmr-zem/)


## Features

- **MCP Architecture**: Standalone, modular servers for domain logic (NeMo Curator, DataJuicer).
- **Config-Driven**: Define complex pipelines using simple YAML files.
- **ZenML Integration**: automatic tracking, visualization, and caching (optional).
- **Dynamic Visualization**: Steps are labeled dynamically in the ZenML dashboard (e.g., `nemo_pii_removal`).

## Getting Started

### 1. Requirements

- Python 3.10+
- `uv` (recommended)

### 2. Run a Pipeline

```bash
# Run pii removal and cleaning with NeMo Curator
uv run zem run tests/manual/nemo_config.yaml

# Run comprehensive filtering with DataJuicer
uv run zem run tests/manual/data_juicer_config.yaml
```

### 3. Visualize with ZenML

```bash
# Start ZenML server (use allowed port range: 8871-8879)
uv run zenml up --port 8871
```

## Architecture

`Zem` uses the **Model Context Protocol (MCP)**:
- **Servers**: Reside in `src/xfmr_zem/servers/`. Each server exposes tools via standard stdio JSON-RPC.
- **Client**: `PipelineClient` reads your config and executes tools across servers using ZenML steps.

## YAML Configuration

```yaml
name: my_custom_pipeline

servers:
  nemo: src/xfmr_zem/servers/nemo_curator/server.py
  dj: src/xfmr_zem/servers/data_juicer/server.py

pipeline:
  - nemo.pii_removal:
      input:
        data: [{"text": "Hello World"}]
  - dj.clean_html:
      input: {} # Implicitly takes data from pii_removal
```

## Development

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
