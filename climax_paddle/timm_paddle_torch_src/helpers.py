""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman

Extracted from timm 1.0.24 layers/helpers.py (lines 10-33) for the
timm_paddle minimal closure. Pure standard library.
"""
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_ntuple = _ntuple
