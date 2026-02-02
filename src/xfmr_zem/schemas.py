from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, RootModel

class StepInput(BaseModel):
    data: Optional[Any] = None
    model_config = {"extra": "allow"}

class PipelineStep(RootModel):
    root: Union[str, Dict[str, Any]]

class ZemConfig(BaseModel):
    name: str = "dynamic_generated_pipeline"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    servers: Dict[str, str] = Field(default_factory=dict)
    pipeline: List[PipelineStep]
