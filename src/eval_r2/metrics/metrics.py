import numpy as np


class ManualModelFitEvaluator:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, num_params: int):
        self.y_true = y_true
        self.y_pred = y_pred
        self.k = num_params
        self.n = len(y_true)
        self._compute()

    def _compute(self):
        self.residuals = self.y_true - self.y_pred
        self.sse = np.sum(self.residuals**2)
        self.sst = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        self.rv = self.sse / (self.n - self.k)

        # R² and Adjusted R²
        self.rsq = 1 - self.sse / self.sst
        self.rsq_adj = 1 - (1 - self.rsq) * (self.n - 1) / (self.n - self.k - 1)

        # Log-likelihood assuming normal errors
        self.log_likelihood = -0.5 * self.n * (np.log(2 * np.pi * self.rv) + 1)

        # AIC, AICc, BIC
        self.aic = 2 * self.k - 2 * self.log_likelihood
        self.aic_c = self.aic + (2 * self.k * (self.k + 1)) / (self.n - self.k - 1)
        self.bic = self.k * np.log(self.n) - 2 * self.log_likelihood

    def get_stats(self):
        return {
            "rsq": self.rsq,
            "rsq_adj": self.rsq_adj,
            "aic": self.aic,
            "aic_c": self.aic_c,
            "bic": self.bic,
            "log_likelihood": self.log_likelihood,
            "residual_variance": self.rv,
        }

    @staticmethod
    def akaike_weights(aic_c_values: list[float]):
        """
        Compute Akaike weights for a list of AICc values.
        """
        aic_c_array = np.array(aic_c_values)
        delta = aic_c_array - np.min(aic_c_array)
        weights = np.exp(-0.5 * delta)
        return weights / np.sum(weights)
