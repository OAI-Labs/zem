
from typing import Dict, Any, List
import yaml
from pathlib import Path
from zenml import pipeline
from .zenml_wrapper import mcp_generic_step
import os
import sys

from .schemas import ZemConfig
from .zenml_wrapper import mcp_generic_step, list_mcp_tools

class PipelineClient:
    """
    Client to run Zem pipelines using MCP servers and ZenML orchestration.
    """
    def __init__(self, config_path: str, params_path: str = None):
        self.config_path = Path(config_path)
        self.params_path = params_path
        self.params = {} 
        self.config_dict = self._load_config_dict(self.config_path)
        
        # 6. Validate with Pydantic
        self.config = ZemConfig(**self.config_dict)
        self.server_configs = self._load_server_configs()

    def _load_params(self, params_path: str = None) -> Dict[str, Any]:
        """Load parameters from default or specified path."""
        params = {}
        default_params_path = self.config_path.parent / "parameters.yml"
        if default_params_path.exists():
            with open(default_params_path, "r") as f:
                params.update(yaml.safe_load(f) or {})
        
        if params_path:
            p_path = Path(params_path)
            if p_path.exists():
                with open(p_path, "r") as f:
                    params.update(yaml.safe_load(f) or {})
        return params

    def _load_config_dict(self, path: Path) -> Dict[str, Any]:
        """Load YAML config and perform substitution."""
        with open(path, "r") as f:
            raw_content = f.read()
            
        self.params = self._load_params(None)
        preliminary_dict = yaml.safe_load(raw_content) or {}
        internal_params = preliminary_dict.get("parameters", {})
        if internal_params:
            self.params.update(internal_params)
            
        if self.params_path:
            custom_params = self._load_params(self.params_path)
            self.params.update(custom_params)
            
        content = raw_content
        for key, value in self.params.items():
            content = content.replace(f"{{{{ {key} }}}}", str(value))
            content = content.replace(f"{{{{{key}}}}}", str(value))
            
        return yaml.safe_load(content)

    def _load_server_configs(self) -> Dict[str, Any]:
        servers = self.config.servers
        configs = {}
        for name, path_str in servers.items():
            # 1. Try relative to config file (User's project)
            abs_path = (self.config_path.parent / path_str).resolve()
            
            # 2. If it doesn't exist AND starts with "servers/", check internal package
            if not abs_path.exists() and path_str.startswith("servers/"):
                package_root = Path(__file__).parent.resolve()
                abs_path = (package_root / path_str / "server.py").resolve()
            
            # 3. If it still doesn't exist, try relative to project root
            if not abs_path.exists():
                project_root = Path(__file__).parent.parent.parent.resolve()
                abs_path = (project_root / path_str).resolve()
            
            # 4. If it's a directory, append default filename
            if abs_path.exists() and abs_path.is_dir():
                abs_path = (abs_path / "server.py").resolve()
            
            env = os.environ.copy()
            src_path = str(Path(__file__).parent.parent.resolve())
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path

            server_specific_params = {}
            prefix = f"{name}."
            for key, value in self.params.items():
                if key.startswith(prefix):
                    server_specific_params[key[len(prefix):]] = value
                else:
                    server_specific_params[key] = value
            
            env["ZEM_PARAMETERS"] = yaml.dump(server_specific_params)
            configs[name] = {
                "command": sys.executable,
                "args": [str(abs_path)],
                "env": env
            }
        return configs

    def discover_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch tools from all registered servers."""
        all_tools = {}
        for name, cfg in self.server_configs.items():
            all_tools[name] = list_mcp_tools(cfg["command"], cfg["args"], cfg["env"])
        return all_tools

    def run(self):
        """Build and run the ZenML pipeline."""
        pipeline_steps = self.config.pipeline
        server_configs = self.server_configs
        pipeline_name = self.config.name

        @pipeline(name=pipeline_name, enable_cache=False)
        def dynamic_generated_pipeline(pipeline_params: Dict[str, Any]):
            step_outputs = {} 
            last_output = {}
            
            for i, p_step in enumerate(pipeline_steps):
                step_def = p_step.root
                srv = ""
                tool = ""
                tool_args = {}
                step_alias = f"step_{i}"

                if isinstance(step_def, str):
                    srv, tool = step_def.split(".")
                elif isinstance(step_def, dict):
                    # Check for name at top level or inside the tool dict
                    step_alias = step_def.get("name")
                    
                    # Exclude control keywords
                    control_keys = ["name", "cache"]
                    keys = [k for k in step_def.keys() if k not in control_keys]
                    if not keys: continue
                    key = keys[0]
                    
                    if "." not in key:
                        # Might be another control key or misconfig
                        continue
                        
                    srv, tool = key.split(".")
                    
                    if not step_alias:
                        step_alias = step_def[key].get("name")
                    
                    step_alias = step_alias or f"{srv}.{tool}.{i}"
                    tool_args = step_def[key].get("input", {}) or {}

                # Smart Parallelization & DAG Logic:
                # 1. By default, a step is a root (None) unless it has no 'data' input,
                #    in which case it inherits from the previous step (linear chain).
                # 2. If 'data' is a reference ($step), it depends on that specific step.
                
                current_prev_output = None
                has_explicit_data = "data" in tool_args

                if not has_explicit_data:
                    # No data provided? Inherit from the last executed step to keep simple sequences working
                    current_prev_output = last_output
                else:
                    # Data provided? Check if it's a reference or raw data
                    for k, v in list(tool_args.items()):
                        if isinstance(v, str) and v.startswith("$"):
                            target_step = v[1:]
                            if target_step in step_outputs:
                                if k == "data":
                                    current_prev_output = step_outputs[target_step]
                                    del tool_args[k]
                                else:
                                    # Limitation: ZenML doesn't materialize artifacts nested in dicts
                                    print(f"[Warning] Tool argument '{k}' uses a step reference '{v}'. "
                                          "Currently, only the 'data' field supports cross-step dependencies. "
                                          "This value will be passed as a raw string.")
                            else:
                                raise ValueError(f"Step reference '{v}' not found in previous steps. Available: {list(step_outputs.keys())}")
                
                # 3. Adaptive Caching:
                # Check for 'cache' at top level
                enable_cache = step_def.get("cache", True) if isinstance(step_def, dict) else True
                
                from zenml import step as zenml_step
                unique_step_name = f"{srv}.{tool}.{i}"
                dynamic_step = zenml_step(
                    mcp_generic_step.entrypoint, 
                    name=unique_step_name,
                    enable_cache=enable_cache
                )
                
                step_output = dynamic_step(
                    server_name=srv,
                    tool_name=tool,
                    server_config=server_configs.get(srv, {}),
                    tool_args=tool_args,
                    previous_output=current_prev_output
                )
                
                step_outputs[step_alias] = step_output
                last_output = step_output
            
            return last_output

        return dynamic_generated_pipeline(pipeline_params=self.params)
