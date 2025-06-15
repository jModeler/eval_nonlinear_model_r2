import numpy as np
from scipy.optimize import minimize


class ModelFitter:
    def __init__(self, model_instance):
        self.model = model_instance
        self.fitted_params = None
        self.result = None

    def _squared_error_loss(self, param_values, param_names, x, y_true, **kwargs):
        param_dict = dict(zip(param_names, param_values))
        try:
            self.model.model(x=x, **param_dict, **kwargs)
            y_pred = self.model.predictions
        except Exception:
            return np.inf
        return np.sum((y_true - y_pred) ** 2)

    def fit(self, x: np.ndarray, y: np.ndarray, initial_params: dict, **kwargs):
        param_names = list(initial_params.keys())
        x0 = list(initial_params.values())

        result = minimize(
            lambda p: self._squared_error_loss(p, param_names, x, y, **kwargs),
            x0=x0,
            method="L-BFGS-B",
        )

        self.result = result
        self.fitted_params = dict(zip(param_names, result.x))
        self.model.model(x=x, **self.fitted_params, **kwargs)
        return self.fitted_params
