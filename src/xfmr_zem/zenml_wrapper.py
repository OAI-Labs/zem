
from typing import Any, Dict, Optional, List
from zenml import step
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import os

import subprocess
import time

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
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=0 
    )

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
                 err = process.stderr.read()
                 raise RuntimeError(f"Server closed connection during init. Stderr: {err}")
            
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
                 err = process.stderr.read()
                 raise RuntimeError(f"Server closed connection during {method}. Stderr: {err}")
            
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
        print(f"Error listing tools: {e}")
        return []


@step
def mcp_generic_step(
    server_name: str,
    tool_name: str,
    server_config: Dict[str, Any],
    tool_args: Dict[str, Any],
    previous_output: Optional[Any] = None
) -> Any:
    """
    A generic ZenML step that executes a tool on an MCP server.
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

    command = server_config.get("command", "python")
    args = server_config.get("args", [])
    env = server_config.get("env", os.environ.copy())
    
    print(f"[{server_name}] Executing tool '{tool_name}'")
    
    try:
        params = {
            "name": tool_name,
            "arguments": tool_args
        }
        result_data = run_mcp_tool(command, args, env, "tools/call", params)
        
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
        
        return output_data
        
    except Exception as e:
        import traceback
        with open("/tmp/zenml_error.log", "w") as f:
            f.write(f"Error executing {server_name}.{tool_name}:\n")
            traceback.print_exc(file=f)
        raise RuntimeError(f"Failed to execute MCP tool {server_name}.{tool_name}: {e}")
