"""Unary transformation classes.

The generic core contract lives in ``kcai_data_sampling_core.api``; the
``Fgsm`` class here wraps the pure step function from
``kcai_data_sampling_fgsm.transformations`` into the unary ``apply`` contract.
"""

from kcai_data_sampling_fgsm.api.transformations.fgsm import Fgsm

__all__ = ["Fgsm"]
