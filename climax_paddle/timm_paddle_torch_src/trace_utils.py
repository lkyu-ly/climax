# Extracted from timm 1.0.24 layers/trace_utils.py (lines 1-5) for the
# timm_paddle minimal closure; only the pure-python fallback of
# torch._assert is kept.
def _assert(condition: bool, message: str):
    assert condition, message
