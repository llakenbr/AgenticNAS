from pydantic import BaseModel, Field
from typing import List, Dict

class ForwardStep(BaseModel):
    out: str = Field(..., description="Name of the output variable")
    op: str = Field(..., description="Layer name or operation to call")
    inputs: List[str] = Field(..., description="List of input variable names")

class LayerSpec(BaseModel):
    type: str = Field(..., description="Layer type, e.g. Conv2d, ReLU")
    params: Dict[str, object] = Field(default_factory=dict, description="Layer parameters")

class ModelSpec(BaseModel):
    name: str = Field(..., description="Model class name")
    input_shape: List[int] = Field(..., description="Input tensor shape, e.g. [1,3,224,224]")
    layers: Dict[str, LayerSpec] = Field(..., description="Dictionary of named layers")
    forward: List[ForwardStep] = Field(..., description="List of forward steps describing data flow")