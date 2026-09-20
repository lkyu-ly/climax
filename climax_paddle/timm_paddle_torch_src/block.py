""" Transformer Block

Extracted from timm 1.0.24 models/vision_transformer.py Block (lines 126-207)
for the timm_paddle minimal closure; _create_attn inlined as a direct
Attention construction (ATTN_LAYERS/DiffAttention/LayerType dead code dropped),
LayerScale dropped (init_values must be None -> nn.Identity), norm_layer
default changed to nn.LayerNorm (timm custom LayerNorm chain dropped),
device/dtype plumbing removed.
"""
from typing import Optional, Type

from torch import nn as nn

from .attention import Attention
from .drop import DropPath
from .mlp import Mlp


class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """Initialize Block.

        Args:
            embed_dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale (unsupported here, must be None).
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super().__init__()
        assert init_values is None, 'LayerScale is not part of the timm_paddle minimal closure'

        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(embed_dim)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_mask: Optional = None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
