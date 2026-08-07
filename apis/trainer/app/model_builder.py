import torch
import torch.nn as nn
import importlib
import yaml

from schemas import ModelSpec

## Load layer map from YAML configuration
def import_class(full_class_string):
    module_path, _, class_name = full_class_string.rpartition('.')
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

with open("layer_map.yaml") as f:
    config = yaml.safe_load(f)

layer_map = {k: import_class(v) for k, v in config.items()}
##

def build_model_class_from_spec(spec: ModelSpec):
    class DynamicModel(nn.Module):
        def __init__(self):
            super().__init__()
            for name, layer_spec in spec.layers.items():
                # Check if layer type is supported
                if layer_spec.type not in layer_map:
                    raise ValueError(f"Unsupported layer type: {layer_spec.type}")
                
                layer_cls = layer_map[layer_spec.type]
                layer = layer_cls(**layer_spec.params)
                setattr(self, name, layer)

        def forward(self, x):
            # Initialize variable environment with input
            vars = {"x": x}
            # Evaluate each forward step as a variable assignment
            for step in spec.forward:
                inputs = [vars[name] for name in step.inputs]
                op = getattr(self, step.op)
                vars[step.out] = op(*inputs)
            # Return last forward step's output
            return vars[spec.forward[-1].out]

    DynamicModel.__name__ = spec.name
    return DynamicModel