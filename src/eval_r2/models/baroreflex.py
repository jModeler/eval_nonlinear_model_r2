import numpy as np
from scipy.special import expit
from eval_r2.models.base import BaseKernel

class BaroreflexModel(BaseKernel):
    def __init__(self, name: str):
        super().__init__(name)
    
    def model(self, b1: float, b2: float, c: float,  d: float, e: float, x: np.array):
        # compute the appropriate kernel
        self.kernel(b1, e, x, kernel_flag="loglinear2") # this will create self.log_linear_2
        self.k1 = self.log_linear_2
        self.kernel(b2, e, x, kernel_flag="loglinear2") 
        self.k2 = self.log_linear_2
        # calculate the exponent terms
        self.exponent_1 = np.exp(self.k1)
        self.exponent_2 = np.exp(self.k2)
        # calculate f
        bh = 2*b1*b2/(np.abs(b1+b2))
        self.kernel(bh, e, x, kernel_flag="loglinear2")
        self.f = expit(-self.log_linear_2)
        # denominator
        self.denominator = 1 + self.f * self.exponent_1 + (1-self.f) * self.exponent_2
        # now calculate the model predictions
        self.predictions = c + (d-c)/self.denominator   
