import numpy as np
import pytest
from scipy.special import expit
from eval_r2.models.baroreflex import BaroreflexModel

@pytest.fixture
def baroreflex_sample_data():
    b1 = 0.5
    b2 = 1.5
    c = 0.3
    d = 1.0
    e = 2.0
    x = np.array([1.0, 2.0, 3.0])
    return b1, b2, c, d, e, x

@pytest.fixture
def baro_model(baroreflex_sample_data):
    b1, b2, c, d, e, x = baroreflex_sample_data
    model = BaroreflexModel("baro_test")
    model.model(b1=b1, b2=b2, c=c, d=d, e=e, x=x)
    return model

def test_model_attributes_exist(baro_model):
    expected_attributes = [
        "k1",
        "k2",
        "exponent_1",
        "exponent_2",
        "f",
        "denominator",
        "predictions"
    ]
    for attr in expected_attributes:
        assert hasattr(baro_model, attr), f"Attribute {attr} is missing"
        
def test_k1_k2_values(baroreflex_sample_data, baro_model):
    b1, b2, _, _, e, x = baroreflex_sample_data
    expected_k1 = b1 * np.log(x) - np.log(e)
    expected_k2 = b2 * np.log(x) - np.log(e)

    np.testing.assert_allclose(baro_model.k1, expected_k1, rtol=1e-6)
    np.testing.assert_allclose(baro_model.k2, expected_k2, rtol=1e-6)

def test_exponent_values(baro_model):
    expected_exponent_1 = np.exp(baro_model.k1)
    expected_exponent_2 = np.exp(baro_model.k2)

    np.testing.assert_allclose(baro_model.exponent_1, expected_exponent_1, rtol=1e-6)
    np.testing.assert_allclose(baro_model.exponent_2, expected_exponent_2, rtol=1e-6)

def test_f_values(baroreflex_sample_data, baro_model):
    b1, b2, _, _, e, x = baroreflex_sample_data
    bh = 2 * b1 * b2 / np.abs(b1 + b2)
    expected_log_linear = bh * np.log(x) - np.log(e)
    expected_f = expit(-expected_log_linear)

    np.testing.assert_allclose(baro_model.f, expected_f, rtol=1e-6)

def test_denominator(baro_model):
    expected = 1 + baro_model.f * baro_model.exponent_1 + (1 - baro_model.f) * baro_model.exponent_2
    np.testing.assert_allclose(baro_model.denominator, expected, rtol=1e-6)

def test_predictions(baroreflex_sample_data, baro_model):
    _, _, c, d, _, _ = baroreflex_sample_data
    expected = c + (d - c) / baro_model.denominator
    np.testing.assert_allclose(baro_model.predictions, expected, rtol=1e-6)