# Zem

**Unified Data Pipeline Framework** combining **ZenML**, **NeMo Curator**, and **DataJuicer** for scalable, config-driven multi-domain data processing.

[🌐 Landing Page & Docs](https://khaihoang.github.io/xfmr-zem/)


## Features

- **MCP Architecture**: Standalone, modular servers for domain logic (NeMo Curator, DataJuicer).
- **Config-Driven**: Define complex pipelines using simple YAML files.
- **Dynamic Parallel DAGs**: Support for independent pipeline branches and step dependencies using `$anchor` syntax.
- **Integrated Profiling**: Built-in data profiling server to track data health and quality.
- **Advanced Sinks**: Seamlessly push processed data to **Hugging Face Hub** and **Vector DBs** (Pinecone, Milvus).
- **Frontier LLM Support**: Connect to **Ollama** or **OpenAI** for smart data masking and classification.
- **ZenML Integration**: Automatic tracking, visualization, and performance metrics for every tool call.
- **Advanced Orchestration**: Custom `ParallelLocalOrchestrator` for true concurrent local execution.

## Getting Started

### 1. Requirements

- Python 3.10+
- `uv` (recommended)

### 2. Setup (Parallel Execution)
To enable true parallel execution on your local machine:
```bash
# Set PYTHONPATH so ZenML find the custom orchestrator
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Switch to the parallel stack
uv run zenml stack set parallel_stack
```

### 3. Run a Pipeline

```bash
# Explore available tools across servers
uv run zem list-tools -c tests/manual/dag_test.yaml

# Run a sample parallel pipeline
uv run zem run tests/manual/parallel_test.yaml

# Run data profiling
uv run zem run tests/manual/profiler_test.yaml
```

### 4. Visualize with ZenML

```bash
# Start ZenML server (use allowed port range: 8871-8879)
uv run zenml up --port 8871
```

## Architecture

`Zem` uses the **Model Context Protocol (MCP)**:
- **Servers**: Reside in `src/xfmr_zem/servers/`. Each server exposes tools via standard stdio JSON-RPC.
- **Client**: `PipelineClient` reads your config and executes tools across servers using ZenML steps.

## Advanced Usage

### Dashboard & Observability
```bash
# Open ZenML UI shortcut
uv run zem dashboard

# Compare two step outputs (Differential analysis)
uv run zem preview <artifact_id_1> --id2 <artifact_id_2>

# Random sampling of data
uv run zem preview <artifact_id> --sample --limit 20
```

### DAG Dependencies
You can reference the output of a previous step by its name:
```yaml
pipeline:
  - name: step_a
    dj.clean_content: ...
  - name: step_b
    nemo.normalize:
      input:
        data: "$step_a" # Parallel resolution
```

### Performance Monitoring
Logs automatically capture tool execution time:
`[dj] Tool 'clean_content' finished in 3.11s`

## Development

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
