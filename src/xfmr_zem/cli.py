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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose/debug logging")
def run(config_file, params, verbose):
    """Run a pipeline from a YAML configuration file"""
    # Configure logging based on verbosity
    if verbose:
        os.environ["ZEM_VERBOSE"] = "1"
        logger.remove()
        logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
        console.print("[bold yellow]Verbose mode enabled - DEBUG logging active[/bold yellow]")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    abs_config = os.path.abspath(config_file)
    console.print(f"[bold green]Starting Pipeline:[/bold green] {abs_config}")
    if params:
        console.print(f"[bold blue]Custom Parameters:[/bold blue] {params}")
    
    try:
        client = PipelineClient(abs_config, params_path=params)
        
        # Dashboard URL (Pre-run)
        try:
            workspace_name = "default"
            try:
                from zenml.client import Client
                zn_client = Client()
                workspace_name = getattr(zn_client, "active_workspace_name", 
                                       getattr(zn_client.active_workspace, "name", "default"))
            except:
                pass
            pre_run_url = f"http://127.0.0.1:8871/projects/{workspace_name}/runs"
            console.print(f"[bold blue]Dashboard URL (Pre-run):[/bold blue] [link={pre_run_url}]{pre_run_url}[/link]")
        except:
            pass

        run_response = client.run()
        
        console.print(f"\n[bold blue]Pipeline Execution Finished![/bold blue]")
        console.print(f"Run Name: [cyan]{run_response.name}[/cyan]")
        console.print(f"Status: [yellow]{run_response.status}[/yellow]")
        
        # ZenML dashboard URL
        try:
            run_id = getattr(run_response, "id", None)
            if run_id:
                workspace_name = "default"
                try:
                    from zenml.client import Client
                    client = Client()
                    # Try to get active workspace name
                    if hasattr(client, "active_workspace_name"):
                        workspace_name = client.active_workspace_name
                    elif hasattr(client, "active_workspace"):
                        workspace_name = client.active_workspace.name
                except:
                    pass
                
                dashboard_url = f"http://127.0.0.1:8871/projects/{workspace_name}/runs/{run_id}/dag"
                console.print(f"Dashboard URL (Run): [link={dashboard_url}]{dashboard_url}[/link]")
        except Exception as e:
            logger.debug(f"Could not generate dashboard URL: {e}")
        
        console.print(f"\n[dim]To visualize this run, ensure ZenML dashboard is running:[/dim]")
        console.print(f"[dim]uv run zenml up --port 8871[/dim]")
        console.print(f"[dim]Or view runs via: zem dashboard[/dim]")
        
    except Exception as e:
        console.print(f"\n[bold red]Pipeline Failed:[/bold red] {e}")
        if os.path.exists("/tmp/zenml_error.log"):
            console.print("\n[bold yellow]Error log snippet (/tmp/zenml_error.log):[/bold yellow]")
            with open("/tmp/zenml_error.log", "r") as f:
                console.print(f.read())


@main.command()
def dashboard():
    """Open the ZenML dashboard."""
    import subprocess
    import webbrowser
    console.print("[bold blue]Checking ZenML Dashboard...[/bold blue]")
    # Default ZenML port used in this project
    url = "http://127.0.0.1:8871"
    console.print(f"URL: [link={url}]{url}[/link]")
    try:
        webbrowser.open(url)
    except Exception as e:
        console.print(f"[yellow]Could not open browser automatically: {e}[/yellow]")


@main.command()
@click.argument("artifact_id")
@click.option("--id2", help="Secondary artifact ID for comparison (diff mode)")
@click.option("--limit", "-n", default=10, help="Number of rows to preview")
@click.option("--sample", is_flag=True, help="Show a random sample instead of the head")
def preview(artifact_id, id2, limit, sample):
    """Preview a ZenML artifact (supports diff mode and sampling)"""
    from zenml.client import Client
    import pandas as pd
    import json
    
    def load_art_df(uid):
        art = Client().get_artifact_version(uid)
        d = art.load()
        if isinstance(d, dict) and "path" in d:
            p = d["path"]
            ext = os.path.splitext(p)[1].lower()
            if ext == ".parquet": return pd.read_parquet(p)
            elif ext == ".csv": return pd.read_csv(p)
            elif ext == ".jsonl":
                with open(p, "r") as f: lines = [json.loads(l) for l in f]
                return pd.DataFrame(lines)
        elif isinstance(d, list): return pd.DataFrame(d)
        elif isinstance(d, pd.DataFrame): return d
        elif isinstance(d, dict): return pd.DataFrame([d])
        return d

    try:
        df1 = load_art_df(artifact_id)
        if df1 is None:
            console.print("[bold red]Error:[/bold red] Artifact is empty or could not be loaded.")
            return

        if not isinstance(df1, pd.DataFrame):
            console.print(f"[bold blue]Artifact Preview (Type: {type(df1).__name__}):[/bold blue]")
            console.print(str(df1))
            return

        if id2:
            df2 = load_art_df(id2)
            if df2 is None:
                console.print("[bold red]Error:[/bold red] Could not load second artifact.")
                return
            
            console.print(f"[bold blue]Comparing Artifacts:[/bold blue] {artifact_id} vs {id2}")
            # Simple column/row diff
            cols1, cols2 = set(df1.columns), set(df2.columns)
            console.print(f"  Rows: {len(df1)} -> {len(df2)} ({len(df2)-len(df1):+d})")
            console.print(f"  Cols: {len(df1.columns)} -> {len(df2.columns)}")
            if cols1 != cols2:
                added = cols2 - cols1
                removed = cols1 - cols2
                if added: console.print(f"  [green]+ Added columns:[/green] {added}")
                if removed: console.print(f"  [red]- Removed columns:[/red] {removed}")
            
            # Show a few sample rows from both for side-by-side or sequential feel
            console.print("\n[bold magenta]Diff Sample (df2):[/bold magenta]")
            df_to_show = df2
        else:
            df_to_show = df1

        if sample and len(df_to_show) > limit:
            preview_df = df_to_show.sample(limit)
        else:
            preview_df = df_to_show.head(limit)

        table = Table(show_header=True, header_style="bold magenta", title=f"Preview ({'Sample' if sample else 'Head'} {limit} rows)")
        for col in preview_df.columns: table.add_column(str(col))
        for _, row in preview_df.iterrows():
            row_values = []
            for val in row:
                v_str = str(val)
                if len(v_str) > 100: v_str = v_str[:97] + "..."
                row_values.append(v_str)
            table.add_row(*row_values)
        console.print(table)
            
    except Exception as e:
        console.print(f"[bold red]Error previewing artifact:[/bold red] {e}")


@main.group()
def ocr():
    """OCR related commands (Installation, etc.)"""
    pass

@ocr.command(name="install")
def ocr_install():
    """Install OCR model weights (ONNX/PTH)"""
    from xfmr_zem.servers.ocr.install_models import main as install_main
    install_main()


if __name__ == "__main__":
    main()

