
from typing import Any, Callable, Dict, List, Optional, Union
import yaml
from pathlib import Path
from fastmcp import FastMCP
import inspect

class ZemServer(FastMCP):
    """
    Base class for Zem MCP Servers.
    Extends FastMCP to support parameter loading and standardized tool registration.
    """
    
    def __init__(
        self,
        name: str,
        Dependencies: Optional[List[str]] = None,
        parameter_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.parameter_file = parameter_file
        self.parameters = {}
        
        # 1. Load from file
        if parameter_file:
            self.load_parameters(parameter_file)
            
        # 2. Override with env params (from PipelineClient)
        import os
        env_params_str = os.environ.get("ZEM_PARAMETERS")
        if env_params_str:
            try:
                env_params = yaml.safe_load(env_params_str)
                if isinstance(env_params, dict):
                    self._merge_parameters(env_params)
            except Exception as e:
                print(f"Error loading ZEM_PARAMETERS: {e}")

    def load_parameters(self, file_path: str) -> Dict[str, Any]:
        """Load parameters from YAML file and merge them."""
        path = Path(file_path)
        if path.exists():
            with open(path, "r") as f:
                file_params = yaml.safe_load(f) or {}
                self._merge_parameters(file_params)
            return self.parameters
        return {}

    def _merge_parameters(self, new_params: Dict[str, Any]):
        """Deep merge and dot-notation expansion for parameters."""
        for key, value in new_params.items():
            if "." in key:
                # Expand "tool.param" to {"tool": {"param": value}}
                parts = key.split(".")
                d = self.parameters
                for part in parts[:-1]:
                    if part not in d or not isinstance(d[part], dict):
                        d[part] = {}
                    d = d[part]
                
                last_part = parts[-1]
                if isinstance(value, dict) and last_part in d and isinstance(d[last_part], dict):
                    self._deep_update(d[last_part], value)
                else:
                    d[last_part] = value
            else:
                # Top level merge
                if isinstance(value, dict) and key in self.parameters and isinstance(self.parameters[key], dict):
                    self._deep_update(self.parameters[key], value)
                else:
                    self.parameters[key] = value

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Helper for deep dictionary update."""
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                self._deep_update(target[k], v)
            else:
                target[k] = v

    # Removed custom tool decorator to fix multiple values for argument 'name' error
    # Inherit directly from FastMCP.tool

    def get_data(self, data: Any) -> List[Dict[str, Any]]:
        """
        Standardized way to get data, supporting both direct lists and file references.
        """
        import os
        from loguru import logger
        if isinstance(data, list):
            return data
        
        if isinstance(data, dict) and "path" in data:
            path = data["path"]
            ext = os.path.splitext(path)[1].lower()
            
            logger.info(f"Server {self.name}: Loading reference {path}")
            if ext == ".jsonl":
                import json
                with open(path, "r", encoding="utf-8") as f:
                    return [json.loads(line) for line in f if line.strip()]
            elif ext == ".csv":
                import pandas as pd
                return pd.read_csv(path).to_dict(orient="records")
            elif ext == ".parquet":
                import pandas as pd
                return pd.read_parquet(path).to_dict(orient="records")
            else:
                raise ValueError(f"Unsupported reference extension: {ext}")
        
        logger.debug(f"Server {self.name}: Data is not list or reference, returning raw: {type(data)}")
        return data

    def save_output(self, data: Any, format: str = "parquet") -> Dict[str, Any]:
        """
        Saves output to a temporary file and returns a reference.
        Prevents large data from being sent over JSON-RPC.
        """
        import uuid
        import os
        from loguru import logger
        
        base_dir = "/tmp/zem_artifacts"
        os.makedirs(base_dir, exist_ok=True)
        
        file_id = str(uuid.uuid4())[:8]
        path = os.path.join(base_dir, f"{self.name}_output_{file_id}.{format}")
        
        logger.info(f"Server {self.name}: Saving result to reference {path}")
        
        if format == "parquet":
            import pandas as pd
            logger.debug(f"Server {self.name}: Converting to DataFrame, data type: {type(data)}")
            try:
                df = pd.DataFrame(data)
                df.to_parquet(path, index=False)
            except Exception as e:
                logger.error(f"Server {self.name}: Failed to create DataFrame from {type(data)}: {e}")
                raise
        elif format == "jsonl":
            import json
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        return {"path": path, "type": format, "size": os.path.getsize(path)}

    def run(self, transport: str = "stdio"):
        """Run the server."""
        super().run(transport=transport, show_banner=False)
