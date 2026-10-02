"""Subclasses of the generic core contracts.

The transformation classes wrap the pure step math from
``kcai_data_sampling_fgsm.transformations`` into the unary ``apply`` contract.
I/O stays out: reads happen in the ``-job`` dataloaders and writes in the
``-job`` outputwriters; everything flows as image batches.
"""
