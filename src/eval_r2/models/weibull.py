import numpy as np
from eval_r2.models.base import BaseKernel

class WeibullModel(BaseKernel):
    def __init__(self, name: str):
        super().__init__(name)

    def model(self, b: float, d: float, e: float, x: np.array, model_flag: str = "W3", c: float = 1.0):
        # compute the appropriate kernel
        self.kernel(b, e, x, kernel_flag="loglinear") # this will create self.log_linear
        # compute the fraction 1/(1 + exp(b*log(x)-e)), since this is common to all model types in this class
        # we're using expit method from scipy to deal with extreme values in the self.log_linear array
        self.exponent = np.exp(-np.exp(self.log_linear))
        # now calculate the model predictions
        if model_flag == "W3":
            self.predictions = d * self.exponent 
        elif model_flag == "W4":
            self.predictions = c + (d-c) * self.exponent
        else:
            raise ValueError(
                "Invalid model_flag. Please provide either 'W3' or 'W5'."
            )