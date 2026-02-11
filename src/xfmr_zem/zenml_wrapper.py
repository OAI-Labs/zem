
from typing import Any, Dict, Optional, List
from zenml import step, log_artifact_metadata
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from loguru import logger
import json
import os

import subprocess
import time

# Import DVC metadata utilities
try:
    from xfmr_zem.utils.dvc_metadata import DVCMetadataExtractor
    DVC_AVAILABLE = True
except ImportError:
    DVC_AVAILABLE = False

# Helper to run async MCP call synchronously
def run_mcp_tool(
    command: str,
    args: list[str],
    env: Dict[str, str],
    method: str,
    params: Dict[str, Any],
    id: int = 1
) -> Any:
    """
    Manually run the MCP server subprocess and call a method via JSON-RPC over stdio.
    """
    cmd = [command] + args
    
    # Forward stderr to sys.stderr for real-time logging in verbose mode
    import sys
    import threading
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=0 
    )
    
    # Start a thread to stream stderr
    def stream_stderr():
        for line in process.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
    
    stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
    stderr_thread.start()


    try:
        # 1. Initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "zem-client", "version": "1.0"}
            }
        }
        process.stdin.write(json.dumps(init_req) + "\n")
        process.stdin.flush()
        
        # Read init response
        while True:
            line = process.stdout.readline()
            if not line:
                 raise RuntimeError(f"Server closed connection during init. Check logs above for details.")
            
            if line.strip().startswith("{"):
                try:
                    json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        
        # 2. Call Method
        message_id = id + 1
        call_req = {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": method,
            "params": params
        }
        process.stdin.write(json.dumps(call_req) + "\n")
        process.stdin.flush()
        
        # Read response
        while True:
            line = process.stdout.readline()
            if not line:
                 raise RuntimeError(f"Server closed connection during {method}. Check logs above for details.")
            
            if line.strip().startswith("{"):
                try:
                    resp = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
             
        # Check for errors
        if "error" in resp:
            raise RuntimeError(f"MCP Protocol Error: {resp['error']}")
            
        result = resp.get("result", {})
        if method == "tools/call" and result.get("isError"):
             err_msg = ""
             if "content" in result:
                 for item in result["content"]:
                     if item.get("type") == "text":
                         err_msg += item.get("text", "")
             raise RuntimeError(f"MCP Tool Error (isError): {err_msg or 'Unknown error'}")

        return result

    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except:
            process.kill()


def list_mcp_tools(
    command: str,
    args: list[str],
    env: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Fetch the list of tools from an MCP server.
    """
    try:
        result = run_mcp_tool(command, args, env, "tools/list", {})
        return result.get("tools", [])
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        return []


@step
def mcp_generic_step(
    server_name: str,
    tool_name: str,
    server_config: Dict[str, Any],
    tool_args: Dict[str, Any],
    previous_output: Optional[Any] = None,
    track_dvc: bool = True
) -> Any:
    """
    A generic ZenML step that executes a tool on an MCP server.
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to execute
        server_config: Server configuration (command, args, env)
        tool_args: Arguments to pass to the tool
        previous_output: Output from previous step (optional)
        track_dvc: Whether to track DVC metadata for data inputs (default: True)
    """
    # Merge previous output into tool_args if present
    if previous_output is not None:
        # If previous_output is a dict and has 'data', it's likely the result of another step
        # In Zem, we usually pass 'data' around.
        
        # Smart Reference Detection
        is_reference = False
        if isinstance(previous_output, dict) and "path" in previous_output:
             is_reference = True
        
        if isinstance(previous_output, dict):
            if is_reference:
                tool_args["data"] = previous_output
            else:
                # Merge fields if it's a regular dict
                for k, v in previous_output.items():
                    if k not in tool_args:
                        tool_args[k] = v
        else:
            tool_args['data'] = previous_output

    # Track DVC metadata for input data paths
    dvc_metadata = {}
    if track_dvc and DVC_AVAILABLE:
        dvc_metadata = _extract_dvc_metadata(tool_args, server_name, tool_name)

    command = server_config.get("command", "python")
    args = server_config.get("args", [])
    env = server_config.get("env", os.environ.copy())
    
    logger.info(f"[{server_name}] Executing tool '{tool_name}'")
    start_time = time.time()
    
    try:
        params = {
            "name": tool_name,
            "arguments": tool_args
        }
        result_data = run_mcp_tool(command, args, env, "tools/call", params)
        execution_time = time.time() - start_time
        logger.info(f"[{server_name}] Tool '{tool_name}' finished in {execution_time:.2f}s")
        
        output_data = {}
        
        if isinstance(result_data, dict) and "content" in result_data:
            content = result_data["content"]
            if isinstance(content, list) and len(content) > 0:
                item = content[0]
                if item.get("type") == "text":
                    text = item.get("text", "")
                    try:
                        output_data = json.loads(text)
                    except:
                        try:
                            import ast
                            output_data = ast.literal_eval(text)
                        except:
                            output_data = {"raw_output": text}
        else:
             output_data = result_data if isinstance(result_data, dict) else {"raw": str(result_data)}
        
        # Log DVC metadata to ZenML artifact
        if track_dvc and dvc_metadata:
            _log_dvc_metadata_to_zenml(dvc_metadata, output_data, server_name, tool_name, execution_time)
        
        return output_data
        
    except Exception as e:
        import traceback
        with open("/tmp/zenml_error.log", "w") as f:
            f.write(f"Error executing {server_name}.{tool_name}:\n")
            traceback.print_exc(file=f)
        raise RuntimeError(f"Failed to execute MCP tool {server_name}.{tool_name}: {e}")


def _extract_dvc_metadata(tool_args: Dict[str, Any], server_name: str, tool_name: str) -> Dict[str, Any]:
    """
    Extract DVC metadata from tool arguments containing data paths.
    
    Detection strategy (in priority order):
      1. Explicit ``dvc_track_paths`` list in tool_args  (user-defined, most reliable)
      2. Heuristic scan of well-known keys with strict path validation
    """
    metadata = {
        "input_data": [],
        "git_commit": DVCMetadataExtractor.get_git_commit(),
        "git_branch": DVCMetadataExtractor.get_git_branch(),
    }
    
    # --- Strategy 1: explicit paths from user ---
    explicit_paths: List[str] = tool_args.pop("dvc_track_paths", [])
    if isinstance(explicit_paths, str):
        explicit_paths = [explicit_paths]
    
    # --- Strategy 2: heuristic scan (fallback) ---
    if not explicit_paths:
        _KNOWN_DATA_KEYS = {"data", "path", "file_path", "input_path", "data_path", "output_path"}
        for key in _KNOWN_DATA_KEYS:
            if key not in tool_args:
                continue
            value = tool_args[key]
            
            # Handle reference dict with path
            if isinstance(value, dict) and "path" in value:
                candidate = value["path"]
            elif isinstance(value, str):
                candidate = value
            else:
                continue
            
            # Strict validation: must look like a file path AND exist on disk
            if not isinstance(candidate, str):
                continue
            if not os.path.sep in candidate and "." not in candidate:
                continue  # plain strings like "hello" are not paths
            if not os.path.exists(candidate):
                continue
            
            explicit_paths.append(candidate)
    
    # --- Build metadata for validated paths ---
    for path in explicit_paths:
        if not os.path.exists(path):
            logger.debug(f"[DVC] Skipping non-existent path: {path}")
            continue
        
        dvc_hash = DVCMetadataExtractor.get_dvc_hash(path)
        file_stats = DVCMetadataExtractor.get_file_stats(path)
        
        metadata["input_data"].append({
            "path": path,
            "dvc_hash": dvc_hash,
            "dvc_tracked": os.path.exists(f"{path}.dvc"),
            "size": file_stats.get("size_human"),
            "type": file_stats.get("type"),
        })
    
    return metadata if metadata["input_data"] else {}


def _log_dvc_metadata_to_zenml(
    dvc_metadata: Dict[str, Any],
    output_data: Dict[str, Any],
    server_name: str,
    tool_name: str,
    execution_time: float
) -> None:
    """
    Log DVC metadata to ZenML artifact metadata for reproducibility tracking.
    """
    try:
        # Build metadata dict for ZenML
        zenml_metadata = {
            "dvc_tracking": {
                "git_commit": dvc_metadata.get("git_commit"),
                "git_branch": dvc_metadata.get("git_branch"),
                "input_data": dvc_metadata.get("input_data", []),
            },
            "execution": {
                "server": server_name,
                "tool": tool_name,
                "duration_seconds": round(execution_time, 2),
            }
        }
        
        # Add output path info if available
        if isinstance(output_data, dict) and "path" in output_data:
            output_hash = DVCMetadataExtractor.get_dvc_hash(output_data["path"])
            zenml_metadata["dvc_tracking"]["output_data"] = {
                "path": output_data["path"],
                "dvc_hash": output_hash,
            }
        
        # Log to ZenML
        log_artifact_metadata(
            metadata=zenml_metadata
        )
        
        # Log summary
        input_hashes = [d.get("dvc_hash", "N/A")[:8] for d in dvc_metadata.get("input_data", [])]
        if input_hashes:
            logger.info(f"[DVC] Tracked input data hashes: {input_hashes}")
        
    except Exception as e:
        logger.debug(f"Could not log DVC metadata to ZenML: {e}")
