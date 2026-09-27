# kcai-data-sampling-fgsm

Adversarial perturbation with the Fast Gradient Sign Method (FGSM) for data
sampling. Adds the `fgsm` transformation (`model_role = "target"`), wired
through the `kcai_data_sampling.transformations` entry-point group. The target
model is the user's: it arrives at run time through the `models:` section and
is never shipped by this package.

`-fgsm` is numpy-only and opt-in: a default workspace install stays torch-free,
and installing this package is an explicit opt-in
(`UV_INDEX= uv sync --group fgsm`).