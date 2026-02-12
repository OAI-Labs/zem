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
@click.option("--port", "-p", default=8878, help="Port to run the configurator on")
def explore(port):
    """Launch the Pipeline Visual Configurator."""
    import uvicorn
    import webbrowser
    import threading
    import time
    from xfmr_zem.ui.backend import app
    
    url = f"http://127.0.0.1:{port}"
    console.print(f"[bold blue]Launching Zem Visual Configurator...[/bold blue]")
    console.print(f"URL: [link={url}]{url}[/link]")
    
    def open_browser():
        time.sleep(1)  # Wait for server to start
        try:
            webbrowser.open(url)
        except Exception as e:
            console.print(f"[yellow]Could not open browser: {e}[/yellow]")
            
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="0.0.0.0", port=port)

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


# =============================================================================
# Subprocess helpers
# =============================================================================

def _run_cmd(cmd: list, cwd=None, check: bool = False):
    """
    Run a subprocess with consistent error handling.
    
    Args:
        cmd: Command and arguments list
        cwd: Working directory
        check: If True, raise on non-zero exit code
        
    Returns:
        CompletedProcess instance (always has .returncode, .stdout, .stderr)
    """
    import subprocess as _sp
    import shutil
    
    # Verify the executable exists before running
    exe = cmd[0]
    if not shutil.which(exe):
        # Return a fake CompletedProcess so callers can check .returncode
        return _sp.CompletedProcess(cmd, returncode=127, stdout="", stderr=f"{exe}: command not found")
    
    result = _sp.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")
    
    return result


@main.group()
def ocr():
    """OCR related commands (Installation, etc.)"""
    pass

@ocr.command(name="install")
def ocr_install():
    """Install OCR model weights (ONNX/PTH)"""
    from xfmr_zem.servers.ocr.install_models import main as install_main
    install_main()


@main.group()
def audio():
    """Audio processing commands (Transcribe & Diarize)"""
    pass


@audio.command()
@click.option("--input", "-i", type=click.Path(exists=True), required=True, help="Path to input audio file")
@click.option("--output", "-o", type=click.Path(), required=True, help="Path to output file")
@click.option("--format", "-f", type=click.Choice(["text", "json", "srt", "all"]), default="text", show_default=True, help="Output format")
@click.option("--timestamps", "-t", is_flag=True, help="Include timestamps in text output")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--padding", type=float, default=0.2, show_default=True, help="Padding collar in seconds")
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda", show_default=True, help="Device for inference")
@click.option("--no-noise-reduction", is_flag=True, help="Disable noise reduction")
@click.option("--no-cleanup", is_flag=True, help="Keep temporary files")
@click.option("--hotwords", type=str, help="Comma-separated list of hotwords")
@click.option("--context-score", type=float, default=2.0, show_default=True, help="Context score boost for hotwords")
def transcribe(input, output, format, timestamps, verbose, padding, device, no_noise_reduction, no_cleanup, hotwords, context_score):
    """Transcribe audio file with speaker diarization (Vietnamese)"""
    try:
        from xfmr_zem.servers.audio.core.models import PipelineConfig
        from xfmr_zem.servers.audio.pipeline import create_pipeline
    except ImportError as e:
        console.print(f"[bold red]Error importing audio module:[/bold red] {e}")
        console.print("[yellow]Please ensure you have installed the 'voice' optional dependencies:[/yellow]")
        console.print("  pip install '.[voice]'")
        sys.exit(1)

    input_path = Path(input)
    output_path = Path(output)

    console.print(f"[bold green]Audio Transcribe & Diarization[/bold green]")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_path}")

    try:
        # Create configuration
        config = PipelineConfig()
        
        # Apply CLI arguments to config
        config.slicer.collar_padding = padding
        config.diarization.device = device
        config.asr.device = device
        config.preprocessing.enable_noise_reduction = not no_noise_reduction
        config.cleanup_temp = not no_cleanup
        config.verbose = verbose
        config.output.format = format
        config.output.include_timestamps = timestamps
        
        if hotwords:
            config.asr.hotwords = [w.strip() for w in hotwords.split(",") if w.strip()]
            config.asr.context_score = context_score
            console.print(f"[cyan]Hotwords enabled:[/cyan] {config.asr.hotwords}")
        
        # Create and run pipeline
        with console.status("[bold green]Running pipeline...[/bold green]"):
            pipeline = create_pipeline(config)
            transcript = pipeline.process(input_path, output_path)
        
        # Print summary
        console.print(f"\n[bold blue]Transcription Summary:[/bold blue]")
        console.print(f"  Total duration: {transcript.total_duration:.1f}s")
        console.print(f"  Speakers: {transcript.speaker_count}")
        console.print(f"  Turns: {len(transcript.turns)}")
        
        # Print transcript preview
        if transcript.turns:
            console.print("\n[bold]Transcript Preview (first 3 turns):[/bold]")
            for turn in transcript.turns[:3]:
                preview = turn.text[:100] + "..." if len(turn.text) > 100 else turn.text
                console.print(f"  [cyan]{turn.speaker_label}[/cyan]: {preview}")
            if len(transcript.turns) > 3:
                console.print(f"  ... and {len(transcript.turns) - 3} more turns")
                
    except Exception as e:
        logger.exception("An error occurred during pipeline execution")
        console.print(f"[bold red]Error during processing:[/bold red] {e}")
        sys.exit(1)


# =============================================================================
# DVC Data Versioning Commands
# =============================================================================

@main.group()
def data():
    """Data versioning commands (DVC integration)"""
    pass


@data.command(name="init")
@click.option("--remote", "-r", default="local", help="Remote storage type: local, minio, gdrive")
@click.option("--remote-url", help="Remote storage URL (e.g., s3://bucket-name)")
def data_init(remote, remote_url):
    """Initialize DVC for data versioning in current project."""
    console.print("[bold blue]🔧 Initializing DVC for Data Versioning...[/bold blue]")
    
    # Check if DVC is installed
    result = _run_cmd(["dvc", "version"])
    if result.returncode == 127:
        console.print("[bold red]Error:[/bold red] DVC not found. Install with: uv tool install dvc --with dvc-s3 --with boto3")
        return
    if result.returncode != 0:
        console.print(f"[bold red]Error:[/bold red] {result.stderr.strip()}")
        return
    console.print(f"[dim]DVC version: {result.stdout.strip().split()[0] if result.stdout else 'unknown'}[/dim]")
    
    # Initialize DVC
    result = _run_cmd(["dvc", "init"])
    if result.returncode == 0:
        console.print("[green]✓[/green] DVC initialized successfully")
    elif "already initialized" in result.stderr.lower():
        console.print("[yellow]⚠[/yellow] DVC already initialized")
    else:
        console.print(f"[red]✗[/red] DVC init failed: {result.stderr}")
        return
    
    # Setup remote
    if remote == "local":
        remote_url = remote_url or "/tmp/dvc-storage"
        r = _run_cmd(["dvc", "remote", "add", "-d", "local", remote_url])
        if r.returncode == 0:
            console.print(f"[green]✓[/green] Local remote configured: {remote_url}")
        else:
            console.print(f"[yellow]⚠[/yellow] Remote config failed: {r.stderr.strip()}")
    elif remote == "minio":
        if not remote_url:
            console.print("[yellow]⚠[/yellow] MinIO remote URL not provided. Configure manually:")
            console.print("  dvc remote add -d minio s3://your-bucket")
            console.print("  dvc remote modify minio endpointurl https://your-minio-url")
            console.print("  dvc remote modify minio access_key_id YOUR_KEY")
            console.print("  dvc remote modify minio secret_access_key YOUR_SECRET")
        else:
            r = _run_cmd(["dvc", "remote", "add", "-d", "minio", remote_url])
            if r.returncode == 0:
                console.print(f"[green]✓[/green] MinIO remote configured: {remote_url}")
                console.print("[yellow]Note:[/yellow] Configure credentials with dvc remote modify")
            else:
                console.print(f"[yellow]⚠[/yellow] Remote config failed: {r.stderr.strip()}")
    
    # Create .dvcignore if not exists
    dvcignore_path = Path(".dvcignore")
    if not dvcignore_path.exists():
        dvcignore_content = """# DVC Ignore Patterns
.cache/
__pycache__/
*.pyc
*.tmp
.DS_Store
.idea/
.vscode/
*.log
.zen/
.venv/
venv/
build/
dist/
"""
        dvcignore_path.write_text(dvcignore_content)
        console.print("[green]✓[/green] Created .dvcignore")
    
    console.print("\n[bold green]✅ DVC setup complete![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. Track data: [cyan]zem data add data/[/cyan]")
    console.print("  2. Push to remote: [cyan]zem data push[/cyan]")
    console.print("  3. Pull data: [cyan]zem data pull[/cyan]")


@data.command(name="add")
@click.argument("path", type=click.Path(exists=True))
@click.option("--message", "-m", help="Commit message for git")
def data_add(path, message):
    """Track a file or directory with DVC."""
    console.print(f"[bold blue]📦 Tracking data: {path}[/bold blue]")
    
    # Add to DVC
    result = _run_cmd(["dvc", "add", path])
    if result.returncode == 127:
        console.print("[bold red]Error:[/bold red] DVC not found. Install with: uv tool install dvc --with dvc-s3 --with boto3")
        return
    if result.returncode != 0:
        console.print(f"[bold red]Error:[/bold red] {result.stderr}")
        return
    
    console.print(f"[green]✓[/green] Added to DVC: {path}")
    
    # Show hash info
    try:
        from xfmr_zem.utils.dvc_metadata import DVCMetadataExtractor
        dvc_hash = DVCMetadataExtractor.get_dvc_hash(path)
        if dvc_hash:
            console.print(f"[dim]DVC Hash: {dvc_hash}[/dim]")
    except ImportError:
        console.print("[yellow]Warning:[/yellow] Could not load DVC metadata extractor.")
    
    # Git add the .dvc file (only if inside a git repo)
    is_git = _run_cmd(["git", "rev-parse", "--is-inside-work-tree"]).returncode == 0
    
    if is_git:
        dvc_file = f"{path}.dvc"
        gitignore_file = f"{Path(path).parent}/.gitignore" if Path(path).parent != Path(".") else ".gitignore"
        
        _run_cmd(["git", "add", dvc_file])
        if Path(gitignore_file).exists():
            _run_cmd(["git", "add", gitignore_file])
        
        console.print(f"[green]✓[/green] Staged for git: {dvc_file}")
        
        if message:
            result = _run_cmd(["git", "commit", "-m", message])
            if result.returncode == 0:
                console.print(f"[green]✓[/green] Committed: {message}")
            else:
                err_msg = result.stderr.strip() or result.stdout.strip()
                console.print(f"[yellow]⚠[/yellow] Git commit: {err_msg}")
    else:
        console.print("[dim]Skipping git stage (not a git repository)[/dim]")


@data.command(name="push")
@click.option("--remote", "-r", help="Remote name to push to")
def data_push(remote):
    """Push tracked data to remote storage."""
    console.print("[bold blue]⬆️  Pushing data to remote...[/bold blue]")
    
    cmd = ["dvc", "push"]
    if remote:
        cmd.extend(["-r", remote])
    
    result = _run_cmd(cmd)
    if result.returncode == 127:
        console.print("[bold red]Error:[/bold red] DVC not found.")
        return
    if result.returncode == 0:
        console.print("[green]✓[/green] Data pushed successfully")
        if result.stdout:
            console.print(f"[dim]{result.stdout.strip()}[/dim]")
    else:
        console.print(f"[bold red]Error:[/bold red] {result.stderr}")


@data.command(name="pull")
@click.option("--remote", "-r", help="Remote name to pull from")
def data_pull(remote):
    """Pull tracked data from remote storage."""
    console.print("[bold blue]⬇️  Pulling data from remote...[/bold blue]")
    
    cmd = ["dvc", "pull"]
    if remote:
        cmd.extend(["-r", remote])
    
    result = _run_cmd(cmd)
    if result.returncode == 127:
        console.print("[bold red]Error:[/bold red] DVC not found.")
        return
    if result.returncode == 0:
        console.print("[green]✓[/green] Data pulled successfully")
        if result.stdout:
            console.print(f"[dim]{result.stdout.strip()}[/dim]")
    else:
        console.print(f"[bold red]Error:[/bold red] {result.stderr}")


@data.command(name="status")
def data_status():
    """Show DVC data status and tracked files."""
    console.print("[bold blue]📊 DVC Data Status[/bold blue]\n")
    
    # Show remotes
    result = _run_cmd(["dvc", "remote", "list"])
    if result.returncode == 127:
        console.print("[bold red]Error:[/bold red] DVC not found.")
        return
    if result.returncode == 0 and result.stdout.strip():
        console.print("[bold]Configured Remotes:[/bold]")
        for line in result.stdout.strip().split("\n"):
            console.print(f"  • {line}")
    else:
        console.print("[yellow]No remotes configured[/yellow]")
    
    console.print()
    
    # Show status
    result = _run_cmd(["dvc", "status"])
    if result.returncode == 0:
        if result.stdout.strip():
            console.print("[bold]Data Status:[/bold]")
            console.print(result.stdout)
        else:
            console.print("[green]✓[/green] All data is up to date")
    else:
        console.print(f"[dim]{result.stderr.strip()}[/dim]")
    
    # Show tracked files
    console.print("\n[bold]Tracked Data Files:[/bold]")
    dvc_files = list(Path(".").rglob("*.dvc"))
    if dvc_files:
        try:
            from xfmr_zem.utils.dvc_metadata import DVCMetadataExtractor
            
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("File")
            table.add_column("Hash")
            table.add_column("Size")
            
            for dvc_file in dvc_files:
                if dvc_file.name == ".dvc":
                    continue
                data_path = str(dvc_file)[:-4]  # Remove .dvc extension
                dvc_hash = DVCMetadataExtractor.get_dvc_hash(data_path) or "N/A"
                stats = DVCMetadataExtractor.get_file_stats(data_path)
                size = stats.get("size_human", "N/A")
                
                # Truncate hash for display
                hash_display = dvc_hash[:12] + "..." if len(dvc_hash) > 15 else dvc_hash
                table.add_row(data_path, hash_display, size)
            
            console.print(table)
        except ImportError:
            console.print("[yellow]Warning:[/yellow] Could not load DVC metadata extractor.")
    else:
        console.print("[dim]No data files tracked yet[/dim]")


@data.command(name="lineage")
@click.argument("path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def data_lineage(path, as_json):
    """Show data lineage and metadata for a tracked file."""
    try:
        from xfmr_zem.utils.dvc_metadata import DVCMetadataExtractor
        import json as json_lib
        
        lineage = DVCMetadataExtractor.create_lineage_metadata(path)
        
        if as_json:
            console.print(json_lib.dumps(lineage, indent=2, default=str))
        else:
            console.print(f"[bold blue]📋 Data Lineage: {path}[/bold blue]\n")
            
            # DVC Info
            console.print("[bold]DVC Information:[/bold]")
            dvc_info = lineage.get("dvc", {})
            console.print(f"  Hash: [cyan]{dvc_info.get('hash', 'N/A')}[/cyan]")
            console.print(f"  Tracked: {'✓' if dvc_info.get('tracked') else '✗'}")
            if dvc_info.get("remote"):
                console.print(f"  Remotes: {dvc_info['remote']}")
            
            # Git Info
            console.print("\n[bold]Git Information:[/bold]")
            git_info = lineage.get("git", {})
            console.print(f"  Commit: [cyan]{git_info.get('commit', 'N/A')[:12] if git_info.get('commit') else 'N/A'}[/cyan]")
            console.print(f"  Branch: {git_info.get('branch', 'N/A')}")
            
            # Data Info
            console.print("\n[bold]Data Information:[/bold]")
            data_info = lineage.get("data", {})
            console.print(f"  Type: {data_info.get('type', 'N/A')}")
            console.print(f"  Size: {data_info.get('size_human', 'N/A')}")
            if data_info.get("file_count"):
                console.print(f"  Files: {data_info['file_count']}")
            
            console.print(f"\n[dim]Timestamp: {lineage.get('timestamp', 'N/A')}[/dim]")
    except ImportError:
        console.print("[bold red]Error:[/bold red] Could not load DVC metadata extractor utility.")



if __name__ == "__main__":
    main()

