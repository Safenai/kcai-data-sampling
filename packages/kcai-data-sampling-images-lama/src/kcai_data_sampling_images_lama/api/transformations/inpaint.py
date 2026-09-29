"""The ``inpaint`` transformation class.

Wraps the region-mask math
(:func:`kcai_data_sampling_images_lama.transformations.inpaint.build_region_mask`)
into the unary transformation contract: each sample's window is erased and the
tool model fills it with content that was not in the image.
"""

from typing_extensions import override

from kcai_data_sampling_core.api.roles import check_output
from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np

from kcai_data_sampling_images_lama.configs import InpaintTransformationConfig
from kcai_data_sampling_images_lama.transformations.inpaint import build_region_mask


class Inpaint(UnaryTransformation):
    """Erase a rectangle of each image and let a tool model fill it in.

    ``top``/``left`` default to 0; ``height``/``width`` are required (a
    degenerate empty region is refused). The replacement is invented by the
    ``tool_model``, so the map is not reversible: what was in the region is
    gone. ``apply`` is deterministic (``stochastic`` not set), so the seed
    records ``None``.
    """

    algorithm = "inpaint"

    #: The registered config schema this algorithm validates against.
    Config = InpaintTransformationConfig

    #: The model slot this algorithm fills, and the methods it must expose.
    model_role = "tool"
    model_methods = ("inpaint",)

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, h, w, c) uint8
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Erase the configured rectangle and fill it via the tool model.

        Args:
            xs: Batch of image arrays shaped ``(B, H, W, C)``.
            rngs: Unused for this deterministic operation.

        Returns:
            A batch of the same shape; the masked rectangle is rewritten from
            the tool model's output, every other pixel returns untouched.

        Raises:
            ValueError: If the region leaves the frame, or the tool model's
                output breaks the numeric contract.
        """
        del rngs
        masks = build_region_mask(xs.shape, **self.params)
        return check_output(
            self.algorithm,
            self.tool_model,
            "inpaint",
            xs,
            self.tool_model.inpaint(xs, masks),
        )
