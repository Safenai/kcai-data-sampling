"""FGSM, adversarial.

One signed step up the gradient of the target model's loss, bounded in L∞ by
``epsilon``. The loss belongs to the model adapter.
"""

import numpy as np
from typing_extensions import override

from kcai_data_sampling.api.unary import UnaryTransformation


class FGSM(UnaryTransformation):
    """Params: ``epsilon``. Requires ``target_model`` with ``grad(xs)``."""

    algorithm = "fgsm"
    model_role = "target"
    reversible = False

    @override
    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        return xs + self.params["epsilon"] * np.sign(self.target_model.grad(xs))
