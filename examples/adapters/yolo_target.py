"""A runnable starting point for a user's target model: YOLOv8 (ultralytics).

The target model is the **user's**. This adapter shows the
``TargetModel.grad`` contract on normalized batches and exercises one slice of
it for real: a forward pass through ultralytics' underlying ``nn.Module`` under
``torch.enable_grad()``, then the gradient of one stand-in loss (the mean
spatial feature activity). **Replace the model and the loss with yours** — the
recognizer, the loss, and the preprocessing you want gradients against; the
framework code around them stays.

The job itself is numpy-only: ``ultralytics`` and ``torch`` are imported lazily,
only when ``grad`` runs, and they must live in the kernel/venv *you* drive the
job with — never a dependency of the package.

Only ``grad(xs)`` has to exist (``TargetModel``): the batch arrives normalized
to ``[0, 1]`` (see ``kcai_data_sampling_core.api.roles.TargetModel.grad``);
the returned gradient is in the same normalized units, same shape, finite. Only
its sign takes part in the FGSM step.

Config to point at it (see ``examples/config/walkthrough-adversarial.yaml``):

.. code-block:: yaml

    models:
      yolo:
        type: python                # the file route: the adapter is user source
        path: examples/adapters/yolo_target.py
        export: YoloTarget
        weights: yolov8n.pt         # your weights, fetched with `ultralytics` yourself
        params: { conf: 0.25 }

"""

import numpy as np


class YoloTarget:
    """A YOLOv8 target whose ``grad`` is the gradient of a stand-in loss.

    Args:
        weights: Path to the ``.pt`` weights, e.g. ``yolov8n.pt`` (fetched by
            the user via ``ultralytics``, never by the package).
        conf: Detection confidence threshold; passed through to ``YOLO``.
    """

    name = "yolo"

    def __init__(self, weights: str | None = None, conf: float = 0.25) -> None:
        self.weights = weights
        self.conf = conf
        self._model = None

    def _model_lazy(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy: only needed when grad() actually runs

            self._model = YOLO(self.weights)
        return self._model

    def grad(self, xs: np.ndarray) -> np.ndarray:
        """Return the gradient of the stand-in loss with respect to ``xs``.

        Args:
            xs: Batch ``(B, H, W, C)`` in ``[0, 1]`` (integer batch divided by
                its dtype maximum); only the RGB channels are fed to the model.

        Returns:
            The gradient ``(B, H, W, C)``, float, finite — the alpha channel
            gets zero (nothing in the model depends on it).

        Raises:
            ImportError: If ``torch`` or ``ultralytics`` are not importable in
                the driving environment (the job env is numpy-only).
        """
        import torch  # lazy: worker-side, never a job dependency

        from ultralytics import YOLO  # noqa: F401  (ensures torch is the one ultralytics wants)

        model = self._model_lazy()

        rgb = xs[..., :3].astype(np.float32)
        batch = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()
        batch.requires_grad_(True)
        torch.set_grad_enabled(True)

        # The backbone/neck without the detection head, which ultralytics freezes
        # under no_grad at inference: this keeps an autograd graph to the input.
        features = model.model.model[:-1](batch)
        features = features if isinstance(features, (list, tuple)) else [features]
        loss = sum(feature.float().mean() for feature in features)

        dx: torch.Tensor = torch.autograd.grad(loss, batch)[0]  # (B, 3, H, W)
        result = np.zeros(xs.shape, dtype=np.float32)
        result[..., :3] = dx.detach().cpu().numpy().transpose(0, 2, 3, 1)
        result = result / max(float(result.max()), float(-result.min()), 1e-8)
        return result  # sign direction only matters for the step