"""The shared engine behind every albumentations-backed transformation.

One class carries all runtime behaviour (integration plan §7): it wraps a
single bare upstream ``BasicTransform`` — constructed at init from the
resolved parameters — into the unary contract. The generated transformation
classes under ``api/transformations/`` are shells that only name the upstream
transform, their config schema and the enum mappings; everything that could
go wrong lives here, once.

Three contract moves, all measured against the pinned upstream commit:

- **Per-row determinism.** Albumentations is stateful per instance: two calls
  without re-seeding differ. The workspace seeds one generator per row from
  ``[seed, sha1(row_id, params)]``, so the engine re-seeds the upstream
  transform from the row's generator before each single-image call. A row's
  output is then byte-identical whatever batch it travels in and whatever
  order it is processed in — the batch-invariance contract.
- **RGBA.** Image-only transforms either hard-fail on 4 channels or would
  corrupt the alpha plane, so the engine splits the alpha, transforms the RGB
  and re-attaches the alpha untouched. Dual transforms (geometry, flips)
  receive the full 4-channel row as-is, because they must move the alpha plane
  with the pixels. 3-channel batches bypass the split.
- **Shape.** Tier-A transforms keep ``(H, W)`` at their pinned settings and
  the base class's shape check stands. Tier-B transforms (the crop/pad
  family, tagged ``geometry`` on the generated class) have their output
  restored to the input shape: crop-family results are resized back
  nearest-neighbor, pad-family results are center-cropped back — the same
  move ``crop_resize`` makes, generalized.
"""

import os

# albumentations' import-time update check hits pypi.org (2 s timeout) unless
# this is set; the package __init__ sets it too, this is the belt to that
# brace — a direct ``import ...engine`` in a fresh process stays offline.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from typing import Any, ClassVar  # noqa: E402

import albumentations as A  # noqa: E402, N812
from kcai_data_sampling_core.api.unary import UnaryTransformation  # noqa: E402
import numpy as np  # noqa: E402
from typing_extensions import override  # noqa: E402

#: Albumentations seeds are non-negative 31-bit integers.
_SEED_BOUND = 2**31 - 1


def _resize_nearest(row: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resample of one ``(H, W, C)`` row to ``(height, width)``.

    Used for the crop family's ``resize_back``: the crop discarded rows and
    columns, and the resize restores the sample's space without inventing
    blended values (aspect distortion accepted — it is exactly what
    ``crop_resize`` does).

    Args:
        row: The transformed row, shaped ``(H', W', C)``.
        height: Target height (the input row's, before the transform).
        width: Target width (the input row's, before the transform).

    Returns:
        The row resampled to ``(height, width, C)``, same dtype.
    """
    ys = np.linspace(0, row.shape[0], height, endpoint=False).astype(int)
    xs = np.linspace(0, row.shape[1], width, endpoint=False).astype(int)
    return np.asarray(row[ys][:, xs])


def _center_crop(row: np.ndarray, height: int, width: int) -> np.ndarray:
    """Center-crop one ``(H, W, C)`` row back to ``(height, width)``.

    Used for the pad family's ``crop_back``: the pad grew the canvas, and the
    crop trims the symmetric overflow, keeping the image content centered.

    Args:
        row: The transformed row, shaped ``(H', W', C)`` with ``H' >= height``
            and ``W' >= width``.
        height: Target height (the input row's, before the transform).
        width: Target width (the input row's, before the transform).

    Returns:
        The centered ``(height, width, C)`` slice, same dtype.
    """
    top = (row.shape[0] - height) // 2
    left = (row.shape[1] - width) // 2
    return row[top : top + height, left : left + width]


class AlbumentationsUnary(UnaryTransformation):
    """One upstream albumentations transform, wrapped into the unary contract.

    Generated subclasses declare identity and wiring only:

    - ``algorithm`` and ``Config`` — the registered type and its schema.
    - ``transform_class`` — the upstream transform class.
    - ``enum_maps`` — per enum parameter, the friendly name → upstream code.
    - ``pinned`` — upstream-only kwargs fixed by the spec table (shape-affecting
      knobs like ``keep_size=True``), not exposed in the config.
    - ``geometry`` — ``"resize_back"`` (crop family), ``"crop_back"`` (pad
      family) or ``None`` (shape-preserving, the default).

    Every such transformation is stochastic — the seed requirement and the
    per-row generators are what make the draws reproducible — and the family
    stays ``"procedural"`` (no model involved).
    """

    stochastic = True

    #: The upstream transform class; set by every generated subclass.
    transform_class: ClassVar[type] = A.NoOp

    #: Friendly name → upstream value, per enum parameter; set by subclasses.
    #: The value is a cv2 code (int) for interpolation/border knobs and the
    #: upstream string for string enums (``compression_type``, ``mode``,
    #: ``position``), so the mapping values are heterogeneous.
    enum_maps: ClassVar[dict[str, dict[str, Any]]] = {}

    #: Upstream-only kwargs fixed by the spec table; set by subclasses.
    pinned: ClassVar[dict[str, Any]] = {}

    #: Shape restore mode for the crop/pad family; set by subclasses.
    geometry: ClassVar[str | None] = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Build the upstream transform from the resolved parameters.

        Args:
            config: The transformation config: ``seed`` (required — the base
                class enforces it for stochastic algorithms) plus the
                algorithm's own parameters, validated by ``Config``.
        """
        super().__init__(config)
        kwargs: dict[str, Any] = dict(self.params)
        for name, mapping in self.enum_maps.items():
            kwargs[name] = mapping[kwargs[name]]
        kwargs.update(self.pinned)
        self._transform = self.transform_class(**kwargs)
        # Image-only ⇒ split the alpha; dual transforms must move the alpha
        # plane with the pixels, so they receive the 4-channel row as-is.
        self._image_only = isinstance(self._transform, A.ImageOnlyTransform)

    @override
    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None = None) -> np.ndarray:  # (b, h, w, c) uint8
        """Map ``(B, H, W, C)`` → ``(B, H, W, C)`` through the upstream transform.

        Args:
            xs: Batch of image arrays shaped ``(B, H, W, C)`` uint8.
            rngs: One generator per row (required: this algorithm is
                stochastic; the base class provides them per call).

        Returns:
            The transformed batch, same shape and dtype.

        Raises:
            ValueError: If ``rngs`` is missing (a stochastic transformation
                cannot honour the batch-invariance contract without per-row
                generators).
        """
        if rngs is None:
            raise ValueError(
                f"{self.algorithm}: rngs are required — every albumentations-backed transformation "
                "is stochastic and seeds the upstream transform per row"
            )
        height, width = xs.shape[1], xs.shape[2]
        rows = [self._apply_row(row, rng, height, width) for row, rng in zip(xs, rngs, strict=True)]
        return np.stack(rows)

    def _apply_row(self, row: np.ndarray, rng: np.random.Generator, height: int, width: int) -> np.ndarray:
        """Transform one row under its own generator, restoring the shape.

        Args:
            row: One sample, shaped ``(H, W, C)`` uint8.
            rng: The row's generator (from ``[seed, sha1(row_id, params)]``).
            height: The pre-transform height, for the shape restore.
            width: The pre-transform width, for the shape restore.

        Returns:
            The transformed row, ``(height, width, C)`` uint8.
        """
        self._transform.set_random_seed(int(rng.integers(_SEED_BOUND)))
        out_row = self._call(row)
        if self.geometry is None or out_row.shape[:2] == (height, width):
            return out_row
        if self.geometry == "resize_back":
            return _resize_nearest(out_row, height, width)
        return _center_crop(out_row, height, width)

    def _call(self, row: np.ndarray) -> np.ndarray:
        """One single-image upstream call, applying the RGBA rule.

        Args:
            row: One sample, shaped ``(H, W, C)`` uint8.

        Returns:
            The transformed row as the upstream transform returned it.
        """
        if not (self._image_only and row.shape[-1] == 4):
            return np.asarray(self._transform(image=row)["image"])
        rgb, alpha = row[..., :3], row[..., 3:]
        return np.concatenate([self._transform(image=rgb)["image"], alpha], axis=-1)
