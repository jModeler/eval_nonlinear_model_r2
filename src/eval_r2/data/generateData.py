import numpy as np
import inspect


class ModelSimulator:
    def __init__(self, model_instance, x: np.ndarray):
        """
        model_instance: an instantiated model object with a 'model' method
        x: input data array over which to simulate
        """
        self.model = model_instance
        self.x = x
        self.true_values = None
        self.noisy_values = {}

    def simulate(
        self, params: dict, noise_sds=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4], **kwargs
    ):
        """
        params: dict of parameter names to values (must match model.model signature)
        noise_sds: list of std deviations for Gaussian noise to add
        kwargs: other model flags or options to pass to model.model
        """
        sig = inspect.signature(self.model.model)

        # Only check for explicitly named parameters, excluding x, *args, **kwargs
        required_params = [
            p.name
            for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and p.name != "x"  # Exclude x because it's passed separately
        ]

        missing = [p for p in required_params if p not in params]
        if missing:
            raise ValueError(f"Missing parameters for model prediction: {missing}")

        self.model.model(x=self.x, **params, **kwargs)

        if not hasattr(self.model, "predictions") or self.model.predictions is None:
            raise AttributeError(
                "Model instance does not have 'predictions' attribute after calling model()"
            )

        self.true_values = self.model.predictions.copy()

        for sd in noise_sds:
            noise = np.random.normal(0, sd, size=self.true_values.shape)
            self.noisy_values[sd] = self.true_values + noise

        return self.true_values
