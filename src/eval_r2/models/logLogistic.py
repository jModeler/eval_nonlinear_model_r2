import numpy as np
from scipy.special import expit
from eval_r2.models.base import BaseKernel

class LogLogisticModel(BaseKernel):
    def __init__(self, name: str):
        super().__init__(name)

    def model(self, b: float, d: float, e: float, x: np.array, model: str = "L3", f: float = 1.0, c: float = 1.0):
        # compute the appropriate kernel
        self.kernel(b, e, x, kernel_flag="loglinear") # this will create self.log_linear
        # compute the fraction 1/(1 + exp(b*log(x)-e)), since this is common to all model types in this class
        # we're using expit method from scipy to deal with extreme values in the self.log_linear array
        self.sigmoid = expit(-self.log_linear)
        # now calculate the model predictions
        if model == "L3":
            self.predictions = d * self.sigmoid 
        elif model == "L4":
            self.predictions = c + (d-c) * self.sigmoid 
        elif model == "L5":
            self.predictions = c + (d-c) * np.float_power(self.sigmoid, f)
        else:
            raise ValueError(
                "Invalid model. Please provide either 'L3', 'L4' or 'L5'."
            )