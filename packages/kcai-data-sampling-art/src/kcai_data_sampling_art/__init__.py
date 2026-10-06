"""Adversarial attacks from the Adversarial Robustness Toolbox (ART).

Ships the ``art_fgsm`` transformation (``model_role = "target"``): ART's
``FastGradientMethod`` run against the user's target model, bridged by
:class:`kcai_data_sampling_art.estimator.TargetEstimator`. No model framework
is imported and no model ships: the target model is the user's, provided at
run time through the ``models:`` section.
"""

from kcai_data_sampling_art._version_ import __version__ as __version__
