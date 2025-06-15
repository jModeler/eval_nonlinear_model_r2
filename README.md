# R<sup>2</sup> Evaluation for Nonlinear Models

Replication of the paper: "An evaluation of R<sup>2</sup>  as an inadequate measure for nonlinear models in pharmacological and biochemical research: a Monte Carlo approach"

Spiess, A.-N., & Neumeyer, N. (2010). An evaluation of R<sup>2</sup> as an inadequate measure for nonlinear models in pharmacological and biochemical research: A Monte Carlo approach. BMC Pharmacology, 10, 6. https://doi.org/10.1186/1471-2210-10-6 

## Features

* Simulate data using customizable nonlinear models.
* Fit models to data optimizing squared error loss.
* Evaluate model fits with metrics like AICc, BIC, adjusted R², residual variance.
* Support for multiple model variants (L3, L4, L5, B3, B4, B5, W3, W4, Baro5).
* Command line script for full workflow: simulation → fitting → metrics → model selection.
* Progress bars and logging for interactive and batch usage.

---

## Installation

Clone the repository and install dependencies with the `uv` package manager:

```bash
git clone git@github.com:jModeler/eval_nonlinear_model_r2.git
cd eval_nonlinear_model_r2
uv install
```

Make sure you have `uv` installed in your environment. If not, install it with:

```bash
pip install uv
```

---

## Usage

Run the main workflow script:

```bash
uv run src/eval_r2/main.py
```

This will:

1. Simulate data using the Log-Logistic L3 model (defined in the paper).
2. Add Gaussian noise at several levels.
3. Fit all supported model variants to each noisy dataset.
4. Compute model fit statistics.
5. Print best-fitting models per criterion and noise level.

In step 5 above, the model output will look like this:

```bash
...
Best models per criterion for each noise level:

Noise SD = 0.010
  aic: L3 (-6407.5486)
  aic_c: L3 (-6407.5245)
  rsq_adj: Baro5 (0.9981)
  bic: L3 (-6392.8253)
  rsq: Baro5 (0.9981)
  residual_variance: Baro5 (0.0001)
  log_likelihood: Baro5 (3207.0101)

Noise SD = 0.020
  aic: L3 (-4984.5877)
  aic_c: L3 (-4984.5636)
  rsq_adj: Baro5 (0.9922)
  bic: L3 (-4969.8644)
  rsq: Baro5 (0.9922)
  residual_variance: Baro5 (0.0004)
  log_likelihood: Baro5 (2495.5712)

Noise SD = 0.050
  aic: L3 (-3181.1337)
  aic_c: L3 (-3181.1096)
  rsq_adj: L3 (0.9538)
  bic: L3 (-3166.4105)
  rsq: Baro5 (0.9539)
  residual_variance: L3 (0.0024)
  log_likelihood: L3 (1593.5669)

Noise SD = 0.100
  aic: L3 (-1708.9616)
  aic_c: L3 (-1708.9375)
  rsq_adj: L3 (0.8263)
  bic: L3 (-1694.2383)
  rsq: L5 (0.8268)
  residual_variance: L3 (0.0105)
  log_likelihood: L3 (857.4808)

Noise SD = 0.200
  aic: L3 (-390.3281)
  aic_c: L3 (-390.3040)
  rsq_adj: L3 (0.5402)
  bic: L3 (-375.6049)
  rsq: Baro5 (0.5422)
  residual_variance: L3 (0.0394)
  log_likelihood: L3 (198.1641)

Noise SD = 0.400
  aic: L3 (1026.5045)
  aic_c: L3 (1026.5286)
  rsq_adj: L3 (0.2542)
  bic: L3 (1041.2277)
  rsq: Baro5 (0.2565)
  residual_variance: L3 (0.1625)
```

Note how AIC, AICc, BIC fit criteria correctly identify L3 as the best model for the generated data (across all error values), whereas R<sup>2</sup> and R<sup>2</sup><sub>adj.</sub> tend to recommend other models. 

---

## Structure

* `src/eval_r2/models/` — Model class implementations.
* `src/eval_r2/data/` — Data simulation and fitting utilities.
* `src/eval_r2/metrics/` — Model evaluation metrics.
* `src/eval_r2/main.py` — Main script to run full pipeline.

---

## Contributing

Feel free to open issues or submit pull requests for bug fixes, enhancements, or new model support.


