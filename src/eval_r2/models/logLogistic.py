import numpy as np
from base import BaseKernel

class LogLogisticModel(BaseKernel):
    def __init__(self, name: str):
        super().__init__(name)

    def model(self, b: float, d: float, e: float, x: np.array, model: str = "L3", f: float = 1.0, c: float = 1.0):
        # compute the appropriate kernel
        self.kernel(b, e, x, kernel_flag="loglinear") # this will create self.log_linear
        # compute the fraction 1/(1 + exp(b*log(x)-e))
        # To prevent numerical overflow, we'll 
