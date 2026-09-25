"""Unary transformation classes in this package.

The generic core contract lives in ``kcai_data_sampling_core.api``; the class
here (``api.transformations.inpaint.Inpaint``) is the package's own subclass,
wrapping a pure algorithm function from
``kcai_data_sampling_images_lama.transformations`` into the unary ``apply``
contract.
"""

from kcai_data_sampling_images_lama.api.transformations.inpaint import Inpaint

__all__ = ["Inpaint"]