# R<sup>2</sup> Evaluation for Nonlinear Models

Replication of the paper: "An evaluation of R<sup>2</sup>  as an inadequate measure for nonlinear models in pharmacological and biochemical research: a Monte Carlo approach"

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

---

## Structure

* `src/eval_r2/models/` — Model class implementations.
* `src/eval_r2/data/` — Data simulation and fitting utilities.
* `src/eval_r2/metrics/` — Model evaluation metrics.
* `src/eval_r2/main.py` — Main script to run full pipeline.

---

## Contributing

Feel free to open issues or submit pull requests for bug fixes, enhancements, or new model support.


