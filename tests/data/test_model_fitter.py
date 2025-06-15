import numpy as np
import pytest
from eval_r2.models.logLogistic import LogLogisticModel
from eval_r2.data.fitData import ModelFitter


@pytest.fixture
def synthetic_data():
    # Ground truth model parameters
    true_params = {"b": 1.0, "d": 2.0, "e": 5.0}
    x = np.linspace(1, 10, 100)

    # Instantiate and simulate data with noise
    model = LogLogisticModel("true")
    model.model(**true_params, x=x, model_flag="L3")
    y_clean = model.predictions
    y_noisy = y_clean + np.random.normal(0, 0.05, size=len(x))

    return x, y_noisy, true_params


def test_fit_loglogistic_model(synthetic_data):
    x, y, true_params = synthetic_data

    # Create new model instance to fit
    model_to_fit = LogLogisticModel("fit")
    initial_guess = {"b": 0.5, "d": 1.5, "e": 4.0}

    fitter = ModelFitter(model_to_fit)
    fitted_params = fitter.fit(x=x, y=y, initial_params=initial_guess, model_flag="L3")

    assert isinstance(fitted_params, dict)
    assert set(fitted_params.keys()) == {"b", "d", "e"}

    # Ensure predictions were generated
    assert hasattr(model_to_fit, "predictions")
    assert model_to_fit.predictions.shape == y.shape


def test_loss_decreases_during_fit(synthetic_data):
    x, y, true_params = synthetic_data

    model = LogLogisticModel("fit_loss")
    initial_guess = {"b": 0.2, "d": 1.0, "e": 2.0}

    fitter = ModelFitter(model)

    # Compute initial loss
    model.model(**initial_guess, x=x, model_flag="L3")
    initial_loss = np.sum((y - model.predictions) ** 2)

    # Fit model
    fitter.fit(x=x, y=y, initial_params=initial_guess, model_flag="L3")
    final_loss = np.sum((y - model.predictions) ** 2)

    assert final_loss < initial_loss


def test_invalid_model_causes_infinite_loss():
    class BrokenModel:
        def model(self, *args, **kwargs):
            raise RuntimeError("Model failure")

    broken_model = BrokenModel()
    fitter = ModelFitter(broken_model)
    x = np.linspace(1, 10, 10)
    y = np.random.normal(0, 1, size=10)
    initial_guess = {"b": 1.0, "d": 2.0, "e": 5.0}

    loss = fitter._squared_error_loss(
        param_values=list(initial_guess.values()),
        param_names=list(initial_guess.keys()),
        x=x,
        y_true=y,
        model_flag="L3",
    )

    assert loss == np.inf
