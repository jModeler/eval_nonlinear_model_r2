import numpy as np
import pytest
from eval_r2.models.logLogistic import LogLogisticModel
from eval_r2.data.generateData import ModelSimulator


@pytest.fixture
def loglogistic_model_and_data():
    x = np.linspace(1, 10, 100)  # Was 10 before
    model = LogLogisticModel("test_loglogistic")
    return model, x


def test_simulate_basic(loglogistic_model_and_data):
    model, x = loglogistic_model_and_data
    simulator = ModelSimulator(model, x)

    # Only required params for L3
    params = {"b": 1.0, "d": 2.0, "e": 0.5, "model_flag": "L3"}

    true_vals = simulator.simulate(params)

    log_linear = params["b"] * np.log(x) - params["e"]
    sigmoid = 1 / (1 + np.exp(log_linear))
    expected = params["d"] * sigmoid

    np.testing.assert_allclose(true_vals, expected, rtol=1e-5)
    assert np.allclose(simulator.true_values, expected)

    assert len(simulator.noisy_values) > 0
    for sd, noisy_vals in simulator.noisy_values.items():
        assert noisy_vals.shape == expected.shape
        noise = noisy_vals - expected
        assert np.isclose(np.std(noise), sd, rtol=0.2)  # Allow 20% relative tolerance


def test_missing_parameters(loglogistic_model_and_data):
    model, x = loglogistic_model_and_data
    simulator = ModelSimulator(model, x)

    # Missing 'd'
    params = {"b": 1.0, "e": 0.5, "model_flag": "L3"}

    with pytest.raises(ValueError, match="Missing parameters"):
        simulator.simulate(params)


def test_no_predictions_after_model_call(loglogistic_model_and_data):
    model, x = loglogistic_model_and_data

    # Define dummy model method with correct signature but no predictions
    def bad_model(self, b, d, e, x, model_flag="L3"):
        pass  # Missing predictions attribute on purpose

    import types
    model.model = types.MethodType(bad_model, model)

    simulator = ModelSimulator(model, x)
    params = {"b": 1.0, "d": 2.0, "e": 0.5}

    with pytest.raises(AttributeError, match="Model instance does not have 'predictions' attribute after calling model()"):
        simulator.simulate(params)


def test_simulate_custom_noise_levels(loglogistic_model_and_data):
    model, x = loglogistic_model_and_data
    simulator = ModelSimulator(model, x)

    params = {"b": 1.0, "d": 2.0, "e": 0.5, "model_flag": "L3"}
    noise_sds = [0.001, 0.05]

    simulator.simulate(params, noise_sds=noise_sds)

    assert set(simulator.noisy_values.keys()) == set(noise_sds)
    for sd in noise_sds:
        noise = simulator.noisy_values[sd] - simulator.true_values
        assert np.abs(np.std(noise) - sd) < 0.02
