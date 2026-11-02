"""Transformation classes owned by this package.

The generic contracts live in ``kcai_data_sampling_core.api``; the classes
here are thin generated shells (``algorithm``, ``Config``, declarations)
around the shared ``engine.AlbumentationsUnary``. Nothing here reads or writes
files: batches flow in and out as ``(B, H, W, C)`` uint8 arrays.
"""
