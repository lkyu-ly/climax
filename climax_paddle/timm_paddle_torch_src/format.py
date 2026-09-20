""" Tensor format enum

Extracted from timm 1.0.24 layers/format.py (lines 6-14) for the
timm_paddle minimal closure.
"""
from enum import Enum
from typing import Union


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


FormatT = Union[str, Format]
