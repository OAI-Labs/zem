"""
CLI for Zem - Unified Data Pipeline Framework (MCP + ZenML)
"""

import os
import click
from rich.console import Console
from rich.table import Table
from loguru import logger
import sys
from pathlib import Path

from xfmr_zem.client import PipelineClient

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Zem CLI - ZenML + MCP (NeMo Curator & DataJuicer)"""
    pass


@main.command()
def info():
    """Show framework information"""
    console.print("[bold blue]Zem: Unified Data Pipeline Framework[/bold blue]")
    console.print("Version: 0.1.0")
    console.print("\nArchitecture: [green]Model Context Protocol (MCP) + ZenML[/green]")
    console.print("\nIntegrations:")
    console.print("  - [bold]ZenML[/bold]: Orchestration, Visualization & Artifact Tracking")
    console.print("  - [bold]MCP Servers[/bold]: Standalone units for domain-specific logic")
    console.print("  - [bold]NeMo Curator[/bold]: NVIDIA's high-performance curation")
    console.print("  - [bold]DataJuicer[/bold]: Comprehensive data processing operators")


@main.command(name="list-tools")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file to discover tools from")
def list_tools(config):
    """List available MCP tools dynamically from servers"""
    if not config:
        # Fallback to hardcoded list if no config provided (legacy behavior)
        console.print("[yellow]Hint: Provide a config file to see dynamic tool list: zem list-tools -c your_config.yaml[/yellow]")
        _print_static_operators()
        return

    try:
        client = PipelineClient(config)
        all_tools = client.discover_tools()
        
        for srv_name, tools in all_tools.items():
            console.print(f"\n[bold magenta]{srv_name} Server Tools:[/bold magenta]")
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Tool Name")
            table.add_column("Description")
            
            for tool in tools:
                table.add_row(tool.get("name", "N/A"), tool.get("description", "No description"))
            console.print(table)
            
    except Exception as e:
        console.print(f"[bold red]Error discovering tools:[/bold red] {e}")


@main.command()
@click.argument("project_name")
def init(project_name: str):
    """Bootstrap a new Zem project structure."""
    base_path = Path(project_name)
    if base_path.exists():
        console.print(f"[bold red]Error:[/bold red] Path '{project_name}' already exists.")
        sys.exit(1)

    # Create directories
    (base_path / "servers").mkdir(parents=True)
    (base_path / "tests/manual").mkdir(parents=True)
    (base_path / "data").mkdir(parents=True)

    # Create sample server
    sample_server_py = """from xfmr_zem.server import ZemServer
from typing import Any, List

# Initialize the sample server
mcp = ZemServer("SampleAgent")

@mcp.tool()
def hello_world(data: Any) -> List[Any]:
    \"\"\"
    A simple tool that adds a 'greeting' field to each record.
    \"\"\"
    dataset = mcp.get_data(data)
    for item in dataset:
        item["greeting"] = "Hello from your standalone Zem project!"
    return dataset

if __name__ == "__main__":
    mcp.run()
"""
    (base_path / "servers" / "sample_server.py").write_text(sample_server_py)

    # Create sample pipeline
    pipeline_yaml = f"""name: {project_name}_pipeline

servers:
  agent: servers/sample_server.py

pipeline:
  - name: my_first_step
    agent.hello_world:
      input:
        data: [{{"text": "Zem is awesome!"}}]
"""
    (base_path / "pipeline.yaml").write_text(pipeline_yaml)

    console.print(f"[bold green]Success![/bold green] Project '{project_name}' initialized.")
    console.print(f"Created standalone sample server: [cyan]{project_name}/servers/sample_server.py[/cyan]")
    console.print(f"Next steps:\n  cd {project_name}\n  zem list-tools -c pipeline.yaml\n  zem run pipeline.yaml")

@main.command()
def operators():
    """List available MCP tools (Static legacy list)"""
    _print_static_operators()

def _print_static_operators():
    # NeMo Curator Tools
    console.print("\n[bold magenta]NeMo Curator Server Tools:[/bold magenta]")
    nemo_table = Table(show_header=True, header_style="bold cyan")
    nemo_table.add_column("Tool Name")
    nemo_table.add_column("Description")
    
    nemo_table.add_row("pii_removal", "Remove PII using NeMo Curator")
    nemo_table.add_row("text_cleaning", "General text cleaning using NeMo Curator")
    console.print(nemo_table)
    
    # DataJuicer Tools
    console.print("\n[bold magenta]DataJuicer Server Tools:[/bold magenta]")
    dj_table = Table(show_header=True, header_style="bold cyan")
    dj_table.add_column("Tool Name")
    dj_table.add_column("Description")
    
    dj_table.add_row("clean_html", "Remove HTML tags")
    dj_table.add_row("clean_links", "Remove URLs/Links")
    dj_table.add_row("fix_unicode", "Normalize Unicode (NFKC)")
    dj_table.add_row("whitespace_normalization", "Clean extra spaces/newlines")
    dj_table.add_row("text_length_filter", "Filter by character length")
    dj_table.add_row("language_filter", "Heuristic-based language filtering")
    dj_table.add_row("document_simhash_dedup", "Simple SimHash-based deduplication")
    console.print(dj_table)
    
    # IO Tools
    console.print("\n[bold magenta]IO Server Tools (File Handling):[/bold magenta]")
    io_table = Table(show_header=True, header_style="bold cyan")
    io_table.add_column("Tool Name")
    io_table.add_column("Description")
    
    io_table.add_row("load_jsonl", "Load data from JSONL file")
    io_table.add_row("save_jsonl", "Save data to JSONL file")
    io_table.add_row("load_csv", "Load data from CSV file")
    io_table.add_row("save_csv", "Save data to CSV file")
    console.print(io_table)
    
    # Profiler Tools
    console.print("\n[bold magenta]Profiler Server Tools:[/bold magenta]")
    prof_table = Table(show_header=True, header_style="bold cyan")
    prof_table.add_column("Tool Name")
    prof_table.add_column("Description")
    
    prof_table.add_row("profile_data", "Generate summary & metrics for input data")
    console.print(prof_table)


@main.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--params", "-p", type=click.Path(exists=True), help="Path to custom parameters.yml")
def run(config_file, params):
    """Run a pipeline from a YAML configuration file"""
    abs_config = os.path.abspath(config_file)
    console.print(f"[bold green]Starting Pipeline:[/bold green] {abs_config}")
    if params:
        console.print(f"[bold blue]Custom Parameters:[/bold blue] {params}")
    
    try:
        client = PipelineClient(abs_config, params_path=params)
        run_response = client.run()
        
        console.print(f"\n[bold blue]Pipeline Execution Finished![/bold blue]")
        console.print(f"Run Name: [cyan]{run_response.name}[/cyan]")
        console.print(f"Status: [yellow]{run_response.status}[/yellow]")
        
        console.print(f"\n[dim]To visualize this run, ensure ZenML dashboard is running:[/dim]")
        console.print(f"[dim]uv run zenml up --port 8871[/dim]")
        console.print(f"[dim]Or view runs via: zem dashboard[/dim]") # Future proofing hint
        
    except Exception as e:
        console.print(f"\n[bold red]Pipeline Failed:[/bold red] {e}")
        if os.path.exists("/tmp/zenml_error.log"):
            console.print("\n[bold yellow]Error log snippet (/tmp/zenml_error.log):[/bold yellow]")
            with open("/tmp/zenml_error.log", "r") as f:
                console.print(f.read())


@main.command()
@click.argument("artifact_id")
@click.option("--limit", "-n", default=10, help="Number of rows to preview")
def preview(artifact_id, limit):
    """Preview a ZenML artifact (supports file references)"""
    from zenml.client import Client
    import pandas as pd
    import json
    
    console.print(f"[bold blue]Fetching Artifact:[/bold blue] {artifact_id}")
    
    try:
        artifact = Client().get_artifact_version(artifact_id)
        data = artifact.load()
        
        df = None
        source_type = "Direct"
        
        # 1. Check if it's a file reference (Zem pattern)
        if isinstance(data, dict) and "path" in data:
            path = data["path"]
            source_type = f"Reference ({data.get('type', 'unknown')})"
            
            if not os.path.exists(path):
                console.print(f"[bold red]Error:[/bold red] Reference file not found: {path}")
                return
            
            ext = os.path.splitext(path)[1].lower()
            if ext == ".parquet":
                df = pd.read_parquet(path)
            elif ext == ".csv":
                df = pd.read_csv(path)
            elif ext == ".jsonl":
                with open(path, "r") as f:
                    lines = [json.loads(line) for line in f]
                df = pd.DataFrame(lines)
            else:
                console.print(f"[bold yellow]Warning:[/bold yellow] Unsupported reference extension {ext}. Displaying raw dict.")
                console.print(data)
                return
        
        # 2. Check if it's direct data (list of dicts or DataFrame)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            console.print(f"[bold yellow]Raw Data (Non-tabular):[/bold yellow]")
            console.print(data)
            return

        if df is not None:
            console.print(f"Source: [green]{source_type}[/green]")
            console.print(f"Total Rows: [cyan]{len(df)}[/cyan] | Total Columns: [cyan]{len(df.columns)}[/cyan]")
            
            preview_df = df.head(limit)
            
            table = Table(show_header=True, header_style="bold magenta", title=f"Preview (First {limit} rows)")
            
            # Use string representation for columns to avoid rich issues with complex types
            for col in preview_df.columns:
                table.add_column(str(col))
                
            for _, row in preview_df.iterrows():
                # Convert values to strings for display, truncate long strings
                row_values = []
                for val in row:
                    val_str = str(val)
                    if len(val_str) > 100:
                        val_str = val_str[:97] + "..."
                    row_values.append(val_str)
                table.add_row(*row_values)
                
            console.print(table)
            
    except Exception as e:
        console.print(f"[bold red]Error previewing artifact:[/bold red] {e}")


if __name__ == "__main__":
    main()

