"""Adversarial perturbation with the Fast Gradient Sign Method.

Ships the ``fgsm`` transformation (``model_role = "target"``): a budgeted
sign-direction step along the gradient of the user's target model, expressed in
normalized pixel units. The transformation class lives under
``api/transformations/``, the pure step math under ``transformations/``, and
the configuration schema at the package root. No model framework is imported
and no model ships: the target model is the user's, provided at run time
through the ``models:`` section.
"""

from kcai_data_sampling_fgsm._version_ import __version__ as __version__
