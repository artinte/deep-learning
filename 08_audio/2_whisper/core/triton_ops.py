

try:
    import triton
    import triton.language as tl
except ImportError:
    raise RuntimeError("triton import failed; try `pip install --pre triton`")
