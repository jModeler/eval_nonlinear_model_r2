import numpy as np
import pytest
from eval_r2.models.base import BaseKernel


class TestBaseKernel:
    @pytest.fixture
    def kernel_obj(self):
        return BaseKernel(name="test_kernel")

    def test_linear_kernel(self, kernel_obj):
        b, e = 2.0, 1.5
        x = np.array([1, 2, 3])
        kernel_obj.kernel(b, e, x, kernel_flag="linear")
        expected = b * x - e
        np.testing.assert_array_almost_equal(kernel_obj.linear, expected)

    def test_loglinear_kernel(self, kernel_obj):
        b, e = 2.0, 1.5
        x = np.array([1, 2, 3])
        kernel_obj.kernel(b, e, x, kernel_flag="loglinear")
        expected = b * np.log(x) - e
        np.testing.assert_array_almost_equal(kernel_obj.log_linear, expected)

    def test_loglinear2_kernel(self, kernel_obj):
        b, e = 2.0, 1.5
        x = np.array([1, 2, 3])
        kernel_obj.kernel(b, e, x, kernel_flag="loglinear2")
        expected = b * np.log(x) - np.log(e)
        np.testing.assert_array_almost_equal(kernel_obj.log_linear_2, expected)

    def test_invalid_kernel_flag(self, kernel_obj):
        b, e = 1.0, 1.0
        x = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Invalid kernel_flag"):
            kernel_obj.kernel(b, e, x, kernel_flag="invalid_flag")
