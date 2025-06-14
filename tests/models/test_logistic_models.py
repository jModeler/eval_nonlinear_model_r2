import numpy as np
import pytest
from eval_r2.models.logistic import LogisticModel

@pytest.fixture
def sample_data():
    # Sample inputs
    b = 1.0
    d = 2.0
    e = 0.5
    x = np.array([1.0, 2.0, 3.0])
    return b, d, e, x

def test_model_L3(sample_data):
    b, d, e, x = sample_data
    model = LogisticModel("test_B3")
    model.model(b=b, d=d, e=e, x=x, model_flag="B3")
    assert hasattr(model, "predictions")
    np.testing.assert_allclose(model.predictions, d * model.sigmoid)

def test_model_L4(sample_data):
    b, d, e, x = sample_data
    c = 0.5
    model = LogisticModel("test_B4")
    model.model(b=b, d=d, e=e, x=x, model_flag="B4", c=c)
    expected = c + (d - c) * model.sigmoid
    np.testing.assert_allclose(model.predictions, expected)

def test_model_L5(sample_data):
    b, d, e, x = sample_data
    c = 0.5
    f = 2.0
    model = LogisticModel("test_B5")
    model.model(b=b, d=d, e=e, x=x, model_flag="B5", c=c, f=f)
    expected = c + (d - c) * np.float_power(model.sigmoid, f)
    np.testing.assert_allclose(model.predictions, expected)

def test_invalid_model_name(sample_data):
    b, d, e, x = sample_data
    model = LogisticModel("test_invalid")
    with pytest.raises(ValueError, match="Invalid model_flag. Please provide either 'B3', 'B4' or 'B5'."):
        model.model(b=b, d=d, e=e, x=x, model_flag="invalid_model")
