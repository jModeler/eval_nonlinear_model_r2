import numpy as np
import logging
from tqdm import tqdm
import pprint

from eval_r2.models.logLogistic import LogLogisticModel
from eval_r2.models.logistic import LogisticModel
from eval_r2.models.weibull import WeibullModel
from eval_r2.models.baroreflex import BaroreflexModel
from eval_r2.data.generateData import ModelSimulator
from eval_r2.data.fitData import ModelFitter
from eval_r2.metrics.metrics import ManualModelFitEvaluator

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ModelFittingLogger")


def simulate_data():
    """Simulate data using LogLogistic L3 model with fixed seed for reproducibility."""
    np.random.seed(42)
    x = np.linspace(1, 20, 1000)
    true_params = {"b": 1.5, "d": 2.0, "e": 5.0}
    simulator = ModelSimulator(LogLogisticModel("L3_sim"), x)
    simulator.simulate(params=true_params, model_flag="L3")
    return x, simulator


def get_model_variants():
    """Return dictionary of model classes with their initial parameter guesses."""
    return {
        "L3": (LogLogisticModel, {"b": 1.0, "d": 1.0, "e": 4.0}),
        "L4": (LogLogisticModel, {"b": 1.0, "d": 1.0, "e": 4.0, "c": 0.5}),
        "L5": (LogLogisticModel, {"b": 1.0, "d": 1.0, "e": 4.0, "c": 0.5, "f": 1.2}),

        "B3": (LogisticModel, {"b": 1.0, "d": 1.0, "e": 4.0}),
        "B4": (LogisticModel, {"b": 1.0, "d": 1.0, "e": 4.0, "c": 0.5}),
        "B5": (LogisticModel, {"b": 1.0, "d": 1.0, "e": 4.0, "c": 0.5, "f": 1.2}),

        "W3": (WeibullModel, {"b": 1.0, "d": 1.0, "e": 4.0}),
        "W4": (WeibullModel, {"b": 1.0, "d": 1.0, "e": 4.0, "c": 0.5}),

        "Baro5": (BaroreflexModel, {"b1": 1.0, "b2": 1.0, "c": 0.5, "d": 2.0, "e": 5.0}),
    }


def fit_models(x, noisy_data, model_variants):
    results = {}
    logger.info("Starting model fitting across noise levels...")

    for sd in tqdm(noisy_data, desc="Noise levels"):
        y_noisy = noisy_data[sd]
        results[sd] = {}

        for flag, (ModelClass, init_params) in tqdm(model_variants.items(), desc=f"Models @ sd={sd:.2f}", leave=False):
            model_instance = ModelClass(name=flag)
            fitter = ModelFitter(model_instance)

            try:
                fitter.fit(x=x, y=y_noisy, initial_params=init_params)
                metrics = ManualModelFitEvaluator(y_true=y_noisy, y_pred=model_instance.predictions, num_params=len(init_params))
                results[sd][flag] = metrics.get_stats()
                logger.info(f"Fit successful: Model={flag}, Noise SD={sd:.2f}")
            except Exception as e:
                logger.warning(f"Fit failed: Model={flag}, Noise SD={sd:.2f}, Error={e}")
                results[sd][flag] = {"error": str(e)}

    return results

def print_best_models_per_criterion(results):
    criteria = ["aic", "aic_c", "rsq_adj", "bic", "rsq", "residual_variance", "log_likelihood"]

    print("\nBest models per criterion for each noise level:")
    for sd, fits in results.items():
        print(f"\nNoise SD = {sd:.3f}")
        for criterion in criteria:
            # Filter out any models with errors or missing criterion
            valid_fits = {m: stats for m, stats in fits.items() if criterion in stats and "error" not in stats}
            if not valid_fits:
                print(f"  {criterion}: No valid fits")
                continue

            if criterion in ["aic", "aic_c", "bic", "residual_variance"]:
                # Lower is better
                best_model = min(valid_fits, key=lambda m: valid_fits[m][criterion])
            else:
                # Higher is better
                best_model = max(valid_fits, key=lambda m: valid_fits[m][criterion])

            best_value = valid_fits[best_model][criterion]
            print(f"  {criterion}: {best_model} ({best_value:.4f})")


def main():
    x, simulator = simulate_data()
    model_variants = get_model_variants()
    results = fit_models(x, simulator.noisy_values, model_variants)
    pprint.pprint(results)
    print_best_models_per_criterion(results)


if __name__ == "__main__":
    main()
