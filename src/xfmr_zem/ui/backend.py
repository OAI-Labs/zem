from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import yaml
from pathlib import Path
from xfmr_zem.client import PipelineClient
from loguru import logger

app = FastAPI(title="Zem Visual Configurator API")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PipelineStep(BaseModel):
    name: Optional[str] = None
    tool: str
    input: Dict[str, Any] = {}
    cache: bool = True

class PipelineConfig(BaseModel):
    name: str
    parameters: Dict[str, Any] = {}
    servers: Dict[str, str] = {}
    pipeline: List[Dict[str, Any]]

@app.get("/api/tools")
async def get_tools(config_path: str = "pipeline.yaml"):
    """Discover all available tools from a sample config."""
    try:
        # We need a dummy config to initialize the client and discover tools
        dummy_path = Path(config_path)
        if not dummy_path.exists():
            # Create a minimal dummy config if none exists
            dummy_content = "name: dummy\nservers:\n  ocr: servers/ocr\n  llm: servers/llm\n  voice: servers/voice\npipeline: []"
            dummy_path.write_text(dummy_content)
        
        client = PipelineClient(str(dummy_path))
        tools = client.discover_tools()
        return tools
    except Exception as e:
        logger.error(f"Error discovering tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-yaml")
async def generate_yaml(config: PipelineConfig):
    """Generate YAML string from the provided configuration."""
    try:
        data = config.dict(exclude_none=True)
        yaml_str = yaml.dump(data, sort_keys=False, allow_unicode=True)
        return {"yaml": yaml_str}
    except Exception as e:
        logger.error(f"Error generating YAML: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-config")
async def save_config(params: Dict[str, Any]):
    """Save the generated YAML to a file."""
    path = params.get("path", "generated_pipeline.yaml")
    yaml_content = params.get("yaml")
    try:
        with open(path, "w") as f:
            f.write(yaml_content)
        return {"status": "success", "path": str(Path(path).absolute())}
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (for production build)
static_dir = Path(__file__).parent / "frontend" / "out"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
else:
    # Fallback to .next/static if using dev mode
    next_dir = Path(__file__).parent / "frontend" / ".next"
    if next_dir.exists():
        logger.warning("Development build detected. For production, run: npm run build in frontend/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
