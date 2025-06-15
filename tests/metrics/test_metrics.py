import numpy as np
import pytest
from eval_r2.metrics.metrics import ManualModelFitEvaluator 


@pytest.fixture
def simple_data():
    y_true = np.array([2.0, 4.0, 6.0, 8.0])
    y_pred = np.array([1.8, 3.9, 6.1, 8.2])
    num_params = 2  # pretend model has 2 fitted parameters
    return y_true, y_pred, num_params


def test_attribute_existence(simple_data):
    y_true, y_pred, k = simple_data
    evaluator = ManualModelFitEvaluator(y_true, y_pred, k)
    for attr in [
        "residuals", "sse", "sst", "rv", "rsq", "rsq_adj",
        "log_likelihood", "aic", "aic_c", "bic"
    ]:
        assert hasattr(evaluator, attr), f"Missing attribute: {attr}"


def test_r_squared(simple_data):
    y_true, y_pred, k = simple_data
    evaluator = ManualModelFitEvaluator(y_true, y_pred, k)

    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    expected_rsq = 1 - sse / sst

    np.testing.assert_allclose(evaluator.rsq, expected_rsq, rtol=1e-6)


def test_adjusted_r_squared(simple_data):
    y_true, y_pred, k = simple_data
    evaluator = ManualModelFitEvaluator(y_true, y_pred, k)

    n = len(y_true)
    rsq = evaluator.rsq
    expected_rsq_adj = 1 - (1 - rsq) * (n - 1) / (n - k - 1)

    np.testing.assert_allclose(evaluator.rsq_adj, expected_rsq_adj, rtol=1e-6)


def test_aic_c_and_bic(simple_data):
    y_true, y_pred, k = simple_data
    evaluator = ManualModelFitEvaluator(y_true, y_pred, k)

    n = len(y_true)
    log_likelihood = evaluator.log_likelihood
    expected_aic = 2 * k - 2 * log_likelihood
    expected_aic_c = expected_aic + (2 * k * (k + 1)) / (n - k - 1)
    expected_bic = k * np.log(n) - 2 * log_likelihood

    np.testing.assert_allclose(evaluator.aic, expected_aic, rtol=1e-6)
    np.testing.assert_allclose(evaluator.aic_c, expected_aic_c, rtol=1e-6)
    np.testing.assert_allclose(evaluator.bic, expected_bic, rtol=1e-6)


def test_get_stats_keys(simple_data):
    y_true, y_pred, k = simple_data
    evaluator = ManualModelFitEvaluator(y_true, y_pred, k)
    stats = evaluator.get_stats()

    expected_keys = {
        "rsq", "rsq_adj", "aic", "aic_c", "bic", "log_likelihood", "residual_variance"
    }
    assert expected_keys.issubset(stats.keys())


def test_akaike_weights():
    aic_c_values = [100.0, 102.0, 105.0]
    weights = ManualModelFitEvaluator.akaike_weights(aic_c_values)

    # Weights should sum to 1
    np.testing.assert_allclose(np.sum(weights), 1.0, rtol=1e-6)

    # The smallest AICc should have the highest weight
    assert weights[0] > weights[1] > weights[2]
