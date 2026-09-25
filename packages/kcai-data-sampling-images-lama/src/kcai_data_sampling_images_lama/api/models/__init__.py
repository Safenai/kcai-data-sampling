"""Tool-model adapters for this package.

Adapters implement the generic ``ToolModel`` protocol from
``kcai_data_sampling_core.api.roles``; the LaMa adapter is the package's first
one. The weight-cache helper lives here too.
"""

from kcai_data_sampling_images_lama.api.models.lama import LamaTool

__all__ = ["LamaTool"]