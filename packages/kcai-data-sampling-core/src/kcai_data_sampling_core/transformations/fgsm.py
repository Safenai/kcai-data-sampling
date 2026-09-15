"""FGSM, adversarial.

One signed step up the gradient of the target model's loss, bounded in L∞ by
``epsilon``. The loss belongs to the model adapter.
"""

import numpy as np
from typing_extensions import override

from kcai_data_sampling_core.api.roles import check_output
from kcai_data_sampling_core.api.unary import UnaryTransformation


class FGSM(UnaryTransformation):
    """One step of ``epsilon``. The target model, the user's, must expose ``grad(xs)``."""

    algorithm = "fgsm"
    model_role = "target"
    model_methods = ("grad",)
    parameters = {"epsilon": None}
    reversible = False

    @override
    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        grad = check_output(self.algorithm, self.target_model, "grad", xs, self.target_model.grad(xs))
        return xs + self.params["epsilon"] * np.sign(grad)
