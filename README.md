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

I also try out this exercise with a sample `xgboost` model, that can be run using the following command:

```bash
uv run src/eval_r2/xgboost_fit.py
```

This should be the output at the end:
```bash
...
Best models per criterion for each noise level:

Noise SD = 0.010
  aic: xgb_d5_eta01 (-36644.5033)
  aic_c: xgb_d5_eta01 (-36579.1538)
  bic: xgb_d4_eta01 (-33114.9311)
  rsq: xgb_d5_eta01 (0.9827)
  rsq_adj: xgb_d5_eta01 (0.9817)
  residual_variance: xgb_d5_eta01 (0.0013)
  log_likelihood: xgb_d5_eta01 (18877.2516)

Noise SD = 0.020
  aic: xgb_d5_eta01 (-34537.5497)
  aic_c: xgb_d5_eta01 (-34472.6837)
  bic: xgb_d4_eta01 (-31339.9578)
  rsq: xgb_d5_eta01 (0.9788)
  rsq_adj: xgb_d5_eta01 (0.9776)
  residual_variance: xgb_d5_eta01 (0.0017)
  log_likelihood: xgb_d5_eta01 (17821.7748)

Noise SD = 0.050
  aic: xgb_d4_eta01 (-26238.8708)
  aic_c: xgb_d4_eta01 (-26217.9138)
  bic: xgb_d4_eta01 (-23945.9825)
  rsq: xgb_d5_eta01 (0.9527)
  rsq_adj: xgb_d5_eta01 (0.9499)
  residual_variance: xgb_d5_eta01 (0.0038)
  log_likelihood: xgb_d5_eta01 (13666.6313)

Noise SD = 0.100
  aic: xgb_d4_eta01 (-15357.7415)
  aic_c: xgb_d4_eta01 (-15336.7846)
  bic: xgb_d3_eta01 (-13798.3071)
  rsq: xgb_d5_eta01 (0.8686)
  rsq_adj: xgb_d5_eta01 (0.8608)
  residual_variance: xgb_d5_eta01 (0.0118)
  log_likelihood: xgb_d5_eta01 (8000.1204)

Noise SD = 0.200
  aic: xgb_d3_eta01 (-2840.4463)
  aic_c: xgb_d3_eta01 (-2835.2100)
  bic: xgb_d3_eta01 (-1686.7918)
  rsq: xgb_d5_eta01 (0.6364)
  rsq_adj: xgb_d4_eta01 (0.6213)
  residual_variance: xgb_d4_eta01 (0.0423)
  log_likelihood: xgb_d4_eta01 (1620.1686)

Noise SD = 0.400
  aic: xgb_d3_eta01 (10815.4305)
  aic_c: xgb_d3_eta01 (10820.6668)
  bic: xgb_d2_eta01 (11936.8748)
  rsq: xgb_d5_eta01 (0.3171)
  rsq_adj: xgb_d3_eta01 (0.2975)
  residual_variance: xgb_d3_eta01 (0.1672)
  log_likelihood: xgb_d3_eta01 (-5247.7152)
```

Note how all the metrics appear to suggest different models (across all error values), however, the information criteria (AIC, AICc, BIC) tend to select the correct model  `xgb_d3_eta01` at specific noise standard deviation values.

That said:
1. I've observed that increasing the learning rate (`eta` hyperparameter) results in all metrics favouring the xgboost models with a higher learning rate (i.e. models that overfit by "learning from the noise")
2. In some cases, a higher tree depth is preferred, even though the true data were generated from a shallow tree
3. Interested users could incorporate (1) cross-validation and (2) regularization hyperparameters to test whether the metrics favor the true model.
4. This might have something to do with the findings that [overparameterized trees tend to fit data better](https://www.pnas.org/doi/10.1073/pnas.1903070116), and do not follow the traditional bias-variance trade off.
5. I use a custom function to count the number of parameters in the xgboost tree. Since this is a parameter used in fit metric calculations, this could be a source of error too.

---

## Structure

* `src/eval_r2/models/` — Model class implementations.
* `src/eval_r2/data/` — Data simulation and fitting utilities.
* `src/eval_r2/metrics/` — Model evaluation metrics.
* `src/eval_r2/main.py` — Main script to run full pipeline.

---

## Contributing

Feel free to open issues or submit pull requests for bug fixes, enhancements, or new model support.


