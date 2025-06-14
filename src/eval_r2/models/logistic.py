import numpy as np
from scipy.special import expit
from eval_r2.models.base import BaseKernel

class LogisticModels(BaseKernel):
    def __init__(self, name: str):
        super().__init__(name)
    
    def model(self, b: float, d: float, e: float, x: np.array, model: str = "B3", f: float = 1.0, c: float = 1.0):
        # compute the appropriate kernel
        self.kernel(b, e, x, kernel_flag="linear") # this will create self.linear
        # compute the fraction 1/(1 + exp(b*log(x)-e)), since this is common to all model types in this class
        # we're using expit method from scipy to deal with extreme values in the self.log_linear array
        self.sigmoid = expit(self.linear)
        # now calculate the model predictions
        if model == "B3":
            self.predictions = d * self.sigmoid 
        elif model == "B4":
            self.predictions = c + (d-c) * self.sigmoid 
        elif model == "B5":
            self.predictions = c + (d-c) * np.float_power(self.sigmoid, f)
        else:
            raise ValueError(
                "Invalid model. Please provide either 'B3', 'B4' or 'B5'."
            )