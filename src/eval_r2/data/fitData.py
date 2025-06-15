import numpy as np
from scipy.optimize import minimize

class ModelFitter:
    def __init__(self, model_instance):
        """
        model_instance: An instantiated model with a `.model()` method and `.predictions` attribute
        """
        self.model = model_instance
        self.fitted_params = None
        self.result = None

    def _squared_error_loss(self, param_values, param_names, x, y_true, **kwargs):
        # Build param dict
        param_dict = dict(zip(param_names, param_values))
        try:
            # Call the model
            self.model.model(x=x, **param_dict, **kwargs)
            y_pred = self.model.predictions
        except Exception as e:
            # Return large loss if model fails
            return np.inf
        
        # Return squared error loss
        return np.sum((y_true - y_pred) ** 2)

    def fit(self, x: np.ndarray, y: np.ndarray, initial_params: dict, **kwargs):
        """
        x: input array
        y: target array
        initial_params: dictionary of initial guesses for parameters
        kwargs: any additional keyword args to pass to the model (e.g., model_flag)
        """
        param_names = list(initial_params.keys())
        x0 = list(initial_params.values())

        # Optimize
        result = minimize(
            self._squared_error_loss,
            x0=x0,
            args=(param_names, x, y, *[], kwargs),
            method='L-BFGS-B'  # Maybe parameterize this to leverage 'Powell', 'Nelder-Mead' for non-smooth problems
        )

        self.result = result
        self.fitted_params = dict(zip(param_names, result.x))

        # Run model once more with fitted params
        self.model.model(x=x, **self.fitted_params, **kwargs)
        return self.fitted_params
