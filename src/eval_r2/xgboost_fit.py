import numpy as np
import xgboost as xgb
import logging
from tqdm import tqdm
from eval_r2.metrics.metrics import ManualModelFitEvaluator
import json

# ----------------------------------------
# Setup logger
# ----------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("XGBoostFittingLogger")


# ----------------------------------------
# Estimate number of parameters
# ----------------------------------------
def estimate_xgboost_num_params(booster: xgb.Booster) -> int:
    """Estimate the number of parameters as the total number of leaf nodes."""
    trees = booster.get_dump(with_stats=True, dump_format='json')

    def count_leaves(node):
        if "leaf" in node:
            return 1
        elif "children" in node:
            return sum(count_leaves(child) for child in node["children"])
        return 0

    total_leaves = 0
    for tree_json in trees:
        tree = json.loads(tree_json)
        total_leaves += count_leaves(tree)

    return total_leaves


# ----------------------------------------
# Simulate data from XGBoost model
# ----------------------------------------
def simulate_xgboost_data(x, noise_sds=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4]):
    """
    Simulate data using predictions from an XGBoost model trained on x with placeholder labels.
    Returns: (true_y, {noise_sd: y_noisy}, true_model_params)
    """
    np.random.seed(42)

    # Step 1: Train model to generate 'true' signal
    placeholder_y = 0.4 * (np.sin(x) + 0.1 * x)  # Arbitrary smooth function, scaled
    dtrain = xgb.DMatrix(x.reshape(-1, 1), label=placeholder_y)
    true_params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": 0.1,
        "verbosity": 0
    }
    booster = xgb.train(true_params, dtrain, num_boost_round=20)
    y_true = booster.predict(dtrain)

    # Step 2: Add noise
    noisy_values = {}
    for sd in noise_sds:
        noise = np.random.normal(0, sd, size=y_true.shape)
        noisy_values[sd] = y_true + noise

    return y_true, noisy_values, true_params


# ----------------------------------------
# Fit variants and compute metrics
# ----------------------------------------
def fit_xgboost_variants(x, noisy_values, variants):
    results = {}
    for sd, y_noisy in tqdm(noisy_values.items(), desc="Noise SDs"):
        results[sd] = {}
        dtrain = xgb.DMatrix(x.reshape(-1, 1), label=y_noisy)

        for name, params in tqdm(variants.items(), desc=f"Variants @ SD={sd:.2f}", leave=False):
            try:
                booster = xgb.train(params, dtrain, num_boost_round=20)
                y_pred = booster.predict(dtrain)

                num_params = estimate_xgboost_num_params(booster)
                evaluator = ManualModelFitEvaluator(y_true=y_noisy, y_pred=y_pred, num_params=num_params)
                results[sd][name] = evaluator.get_stats()
                logger.info(f"Fit success: {name}, SD={sd:.2f}, Params={num_params}")
            except Exception as e:
                logger.warning(f"Fit failed: {name}, SD={sd:.2f}, Error={e}")
                results[sd][name] = {"error": str(e)}

    return results


# ----------------------------------------
# Print best models by fit metric
# ----------------------------------------
def print_best_models_per_criterion(results):
    criteria = ["aic", "aic_c", "bic", "rsq", "rsq_adj", "residual_variance", "log_likelihood"]

    print("\nBest models per criterion for each noise level:")
    for sd, fits in results.items():
        print(f"\nNoise SD = {sd:.3f}")
        for criterion in criteria:
            valid_fits = {
                m: stats for m, stats in fits.items()
                if criterion in stats and "error" not in stats
            }
            if not valid_fits:
                print(f"  {criterion}: No valid fits")
                continue

            if criterion in ["aic", "aic_c", "bic", "residual_variance"]:
                best_model = min(valid_fits, key=lambda m: valid_fits[m][criterion])
            else:
                best_model = max(valid_fits, key=lambda m: valid_fits[m][criterion])

            best_value = valid_fits[best_model][criterion]
            print(f"  {criterion}: {best_model} ({best_value:.4f})")


# ----------------------------------------
# Entrypoint
# ----------------------------------------
def main():
    x = np.linspace(1, 20, 10000)

    y_true, noisy_values, true_params = simulate_xgboost_data(x)

    # Define variants of the XGBoost model
    variants = {
        "xgb_d3_eta01": {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.1, "verbosity": 0},
        "xgb_d4_eta01": {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1, "verbosity": 0},
        #"xgb_d3_eta05": {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.5, "verbosity": 0},
        "xgb_d2_eta01": {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.1, "verbosity": 0},
        "xgb_d5_eta01": {"objective": "reg:squarederror", "max_depth": 5, "eta": 0.1, "verbosity": 0},
    }

    results = fit_xgboost_variants(x, noisy_values, variants)
    print_best_models_per_criterion(results)


if __name__ == "__main__":
    main()
