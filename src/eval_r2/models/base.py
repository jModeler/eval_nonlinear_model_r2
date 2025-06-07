import numpy as np

class BaseKernel:
    def __init__(self, name: str):
        self.name = name

    def kernel(self, b: float, e: float, x: np.array, kernel_flag: str = 'linear'):
        if kernel_flag == 'linear':
            self.linear = b*x - e
        elif kernel_flag == 'loglinear':
            self.log_linear = b*np.log(x) - e 
        elif kernel_flag == 'loglinear2':
            self.log_linear_2 = b*np.log(x) - np.log(e)
        else:
            raise ValueError(
                "Invalid kernel_flag. Please provide either 'linear', 'loglinear' or 'loglinear2'."
            )
    

