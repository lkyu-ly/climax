"""Minimal closure of timm 1.0.24 components needed by the ClimaX backbone.

Produced by paconvert from a torch-side extraction of timm 1.0.24 (see
git history and docs/superpowers/plans/2026-09-20-climax-paddle-port.md
for the extraction table), followed by the manual rewrites documented in
each module.
"""
from .attention import Attention
from .block import Block
from .drop import DropPath, drop_path
from .format import Format
from .helpers import to_1tuple, to_2tuple, to_ntuple
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .trace_utils import _assert
from .weight_init import trunc_normal_

__all__ = [
    "Attention",
    "Block",
    "DropPath",
    "drop_path",
    "Format",
    "to_1tuple",
    "to_2tuple",
    "to_ntuple",
    "Mlp",
    "PatchEmbed",
    "_assert",
    "trunc_normal_",
]
