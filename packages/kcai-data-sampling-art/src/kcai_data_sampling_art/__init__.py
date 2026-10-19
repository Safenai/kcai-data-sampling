"""Adversarial attacks from the Adversarial Robustness Toolbox (ART).

Ships six transformations (``model_role = "target"``), ART attacks run
against the user's target model: the gradient attacks ``art_fgsm``,
``art_pgd`` and ``art_mim``, bridged by
:class:`kcai_data_sampling_art.estimator.TargetEstimator`, and the black-box
attacks ``art_square``, ``art_simba`` and ``art_hopskipjump``, bridged by
:class:`kcai_data_sampling_art.estimator.TargetClassifier`. No model framework
is imported and no model ships: the target model is the user's, provided at
run time through the ``models:`` section.
"""

from kcai_data_sampling_art._version_ import __version__ as __version__
