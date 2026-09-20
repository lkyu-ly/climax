"""Minimal closure of timm 1.0.24 components needed by the ClimaX backbone.

Torch-side extraction; input to paconvert. See timm_paddle/ for the
converted package.
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
    'Attention', 'Block', 'DropPath', 'drop_path', 'Format',
    'to_1tuple', 'to_2tuple', 'to_ntuple', 'Mlp', 'PatchEmbed',
    '_assert', 'trunc_normal_',
]
