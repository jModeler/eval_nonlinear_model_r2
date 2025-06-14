import numpy as np
import pytest
from eval_r2.models.weibull import WeibullModel

@pytest.fixture
def sample_data():
    # Sample inputs
    b = 1.0
    d = 2.0
    e = 0.5
    x = np.array([1.0, 2.0, 3.0])
    return b, d, e, x

def test_model_W3(sample_data):
    b, d, e, x = sample_data
    model = WeibullModel("test_W3")
    model.model(b=b, d=d, e=e, x=x, model="W3")
    assert hasattr(model, "predictions")
    np.testing.assert_allclose(model.predictions, d * model.exponent)

def test_model_W4(sample_data):
    b, d, e, x = sample_data
    c = 0.5
    model = WeibullModel("test_W4")
    model.model(b=b, d=d, e=e, x=x, model="W4", c=c)
    expected = c + (d - c) * model.exponent
    np.testing.assert_allclose(model.predictions, expected)

def test_invalid_model_name(sample_data):
    b, d, e, x = sample_data
    model = WeibullModel("test_invalid")
    with pytest.raises(ValueError, match="Invalid model. Please provide either 'W3' or 'W5'."):
        model.model(b=b, d=d, e=e, x=x, model="invalid_model")
