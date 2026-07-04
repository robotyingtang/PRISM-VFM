from .tp import TransformerProjector
import torch.nn as nn
from dataclasses import asdict
from .options import EncoderOptions, ProjectorOptions
def build_projector(input_dim: int, output_dim: int, extra_args) -> nn.Module:
    args = asdict(ProjectorOptions(input_dim, output_dim))
    # args.update(extra_args)
    return TransformerProjector(**args)

def get_projector(input_dim: int, output_dim: int):
    assert input_dim > 0, input_dim
    assert output_dim > 0, output_dim
    return TransformerProjector(input_dim=input_dim, output_dim=output_dim)
