
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
            if path_str.startswith("servers/"):
                package_root = Path(__file__).parent.resolve()
                abs_path = (package_root / path_str / "server.py").resolve()
            else:
                abs_path = (self.config_path.parent / path_str).resolve()
            
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
                    # Get the key that is not 'name'
                    keys = [k for k in step_def.keys() if k != "name"]
                    if not keys:
                        continue
                    key = keys[0]
                    srv, tool = key.split(".")
                    # Check if it has an alias
                    step_alias = step_def.get("name", f"{srv}.{tool}.{i}")
                    tool_args = step_def[key].get("input", {}) or {}
                
                # If no explicit input provided for 'data', use last_output
                current_prev_output = last_output
                
                # DAG support: Check if input references another step
                # e.g. input: { data: "$step_name" }
                for k, v in list(tool_args.items()):
                    if isinstance(v, str) and v.startswith("$"):
                        target_step = v[1:]
                        if target_step in step_outputs:
                            # If it's the 'data' field, we can pass it as previous_output
                            # which ZenML will materialize properly
                            if k == "data":
                                current_prev_output = step_outputs[target_step]
                                del tool_args[k]
                            else:
                                # For other fields, we still have a limitation unless we add more args
                                tool_args[k] = step_outputs[target_step]
                
                from zenml import step as zenml_step
                unique_step_name = f"{srv}.{tool}.{i}"
                dynamic_step = zenml_step(mcp_generic_step.entrypoint, name=unique_step_name)
                
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
