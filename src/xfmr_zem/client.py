
from typing import Dict, Any, List
import yaml
from pathlib import Path
from zenml import pipeline
from .zenml_wrapper import mcp_generic_step
import os
import sys

class PipelineClient:
    """
    Client to run Zem pipelines using MCP servers and ZenML orchestration.
    """
    def __init__(self, config_path: str, params_path: str = None):
        self.config_path = Path(config_path)
        self.params_path = params_path
        self.params = {} # Will be populated in _load_config
        self.config = self._load_config(self.config_path)
        self.server_configs = self._load_server_configs()

    def _load_params(self, params_path: str = None) -> Dict[str, Any]:
        """Load parameters from default or specified path."""
        params = {}
        
        # Try default parameters.yml in the same directory as config
        default_params_path = self.config_path.parent / "parameters.yml"
        if default_params_path.exists():
            with open(default_params_path, "r") as f:
                params.update(yaml.safe_load(f) or {})
        
        # Load custom parameters if provided
        if params_path:
            p_path = Path(params_path)
            if p_path.exists():
                with open(p_path, "r") as f:
                    params.update(yaml.safe_load(f) or {})
            else:
                print(f"Warning: Parameters file not found: {params_path}")
        
        return params

    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load YAML config, merge parameters (global -> internal -> custom), and perform substitution."""
        with open(path, "r") as f:
            raw_content = f.read()
            
        # 1. Load global defaults (parameters.yml)
        self.params = self._load_params(None)
            
        # 2. Preliminary load to merge internal parameters (priority > global)
        preliminary_dict = yaml.safe_load(raw_content) or {}
        internal_params = preliminary_dict.get("parameters", {})
        if internal_params:
            self.params.update(internal_params)
            
        # 3. Load custom execution overrides (priority > internal)
        if self.params_path:
            custom_params = self._load_params(self.params_path)
            self.params.update(custom_params)
            
        # 4. Perform string substitution for {{ var }}
        content = raw_content
        for key, value in self.params.items():
            content = content.replace(f"{{{{ {key} }}}}", str(value))
            content = content.replace(f"{{{{{key}}}}}", str(value))
            
        # 5. Final load of substituted content
        return yaml.safe_load(content)

    def _load_server_configs(self) -> Dict[str, Any]:
        # Helper to locate server definitions based on the 'servers' block in config
        # Similar to UltraRAG's logic
        servers = self.config.get("servers", {})
        configs = {}
        for name, path_str in servers.items():
            # Support short notation: servers/nemo_curator -> src/xfmr_zem/servers/nemo_curator/server.py
            if path_str.startswith("servers/"):
                package_root = Path(__file__).parent.resolve()
                abs_path = (package_root / path_str / "server.py").resolve()
            else:
                abs_path = (self.config_path.parent / path_str).resolve()
            
            if not abs_path.exists():
                print(f"Warning: Server implementation not found at {abs_path}")

            env = os.environ.copy()
            # Inject src directory into PYTHONPATH so server can import xfmr_zem
            src_path = str(Path(__file__).parent.parent.resolve())
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path

            # Pass custom parameters down to the server
            # Filter parameters meant for this specific server (e.g. starting with "dj.")
            server_specific_params = {}
            prefix = f"{name}."
            
            for key, value in self.params.items():
                if key.startswith(prefix):
                    stripped_key = key[len(prefix):]
                    server_specific_params[stripped_key] = value
                else:
                    # Also pass all root parameters to all servers by default 
                    # for backward compatibility and shared config
                    server_specific_params[key] = value
            
            env["ZEM_PARAMETERS"] = yaml.dump(server_specific_params)

            configs[name] = {
                "command": sys.executable, # Use current python interpreter
                "args": [str(abs_path)],
                "env": env
            }
        return configs

    def run(self):
        """Build and run the ZenML pipeline."""
        
        pipeline_steps = self.config.get("pipeline", [])
        server_configs = self.server_configs
        
        pipeline_name = self.config.get("name", "dynamic_generated_pipeline")

        @pipeline(name=pipeline_name, enable_cache=False)
        def dynamic_generated_pipeline(pipeline_params: Dict[str, Any]):
            # Iterate through steps defined in YAML
            # Data flow tracking is simplified here (linear)
            last_output = {} 
            
            for step_def in pipeline_steps:
                srv = ""
                tool = ""
                tool_args = {}

                if isinstance(step_def, str):
                    # Format: "server.tool"
                    try:
                        srv, tool = step_def.split(".")
                    except ValueError:
                        print(f"Invalid step format: {step_def}")
                        continue
                elif isinstance(step_def, dict):
                    # Format: {"server.tool": {"input": {...}}}
                    key = list(step_def.keys())[0]
                    try:
                        srv, tool = key.split(".")
                        tool_args = step_def[key].get("input", {}) or {}
                    except ValueError:
                        print(f"Invalid step format: {step_def}")
                        continue
                
                if srv and tool:
                    # Dynamically create a step name for better visualization
                    from zenml import step as zenml_step
                    # Use dot notation for step names as requested
                    dynamic_step = zenml_step(mcp_generic_step.entrypoint, name=f"{srv}.{tool}")
                    step_output = dynamic_step(
                        server_name=srv,
                        tool_name=tool,
                        server_config=server_configs.get(srv, {}),
                        tool_args=tool_args,
                        previous_output=last_output
                    )
                    last_output = step_output
            
            return last_output

        # Run the pipeline with parameters to make them visible in ZenML Dashboard
        p = dynamic_generated_pipeline(pipeline_params=self.params)
        
        # Try to inspect output if possible (ZenML returns the pipeline object or execution result)
        # Note: In ZenML 0.x, .run() returns the run metadata, but we can't easily get the data back
        # synchronously here without accessing the artifact store.
        # But for MVP verification, we'll just log success.
        return p
        # p.run() - dynamic_generated_pipeline() already runs/submits the pipeline
