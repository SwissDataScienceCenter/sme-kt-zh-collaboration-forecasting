# SME-KT-ZH Collaboration Forecasting

This project applies **survival analysis** to B2B/B2C sales transaction data to predict when customers are likely to place their next order. The output is a ranked priority list of customers, enabling proactive outreach and collaboration planning.

---

## Table of Contents

- [Data](#data)
- [Notebooks](#notebooks)
- [Source Modules](#source-modules)
- [Installation](#installation)
- [Development Tools](#development-tools)

---

## Data

The `data/` directory contains two files:

| File | Description |
|---|---|
| `sales_df.csv` | Transaction-level sales records with customer IDs, dates, and customer category/type attributes |
| `feiertage.csv` | Swiss public holiday calendar used as an external covariate |

> **Note:** `sales_df` contains **synthetic data**. The data is generated to reflect the statistical properties and patterns of real sales transactions, including realistic customer ordering cadences, seasonal effects, and B2B/B2C customer mix. It does not contain any personal or commercially sensitive information.

---

## Notebooks

Notebooks are located in `notebooks/` and should be run **in the following order**. Each builds on the insights of the previous one.

### 1. `EDA.ipynb` — Exploratory Data Analysis

Start here. This notebook provides a thorough understanding of the data before any modeling is attempted:

- Transaction-level overview and customer segmentation (B2C vs. B2B)
- Temporal patterns: daily, weekly, monthly, and quarterly seasonality
- Multi-seasonal decomposition (MSTL) and stationarity tests (ADF/KPSS)
- Holiday correlation analysis
- AutoGluon time series forecasting benchmarks (with and without holiday covariates)

Running EDA first is essential because survival models are sensitive to data quality and distribution assumptions. Understanding customer ordering cadence, the degree of censoring, and data irregularities directly informs modeling choices.

---

### 2. `Lifelines_Modelling.ipynb` — Parametric Survival Models

Introduces survival analysis via the [`lifelines`](https://lifelines.readthedocs.io) library. Models the time between purchases as a survival problem, where a "next order" is the event and customers who have not yet reordered are right-censored.

Three models are fitted and compared:

| Model | Type | Key characteristic |
|---|---|---|
| **Cox Proportional Hazards (CoxPH)** | Semi-parametric | Makes no assumption about the baseline hazard shape; assumes covariate effects are multiplicative and constant over time (proportional hazards). Highly interpretable. |
| **Weibull AFT** | Parametric | Models time-to-event directly under a Weibull distribution. Assumes a specific hazard shape; covariates stretch or compress the time axis. |
| **Log-Normal AFT** | Parametric | Same AFT framework as Weibull but with a log-normal distribution, allowing for non-monotone hazard. |

**Why start with Lifelines?** These models are interpretable, fast to fit, and provide a strong, explainable baseline. The Cox model in particular has well-understood diagnostics (proportional hazards assumption tests) that help validate whether survival analysis is appropriate for this data. Evaluation uses the concordance index (C-index) and recall@k on a held-out test set.

---

### 3. `RSF_Modelling.ipynb` — Random Survival Forest

Fits a **Random Survival Forest (RSF)** using [`scikit-survival`](https://scikit-survival.readthedocs.io). RSF is a non-parametric ensemble method that:

- Makes no distributional assumptions about the event time
- Captures non-linear covariate effects and feature interactions automatically
- Supports richer feature engineering (recency, frequency, customer category dummies)
- Includes hyperparameter tuning

**Why RSF after Lifelines?** RSF is more complex and less interpretable than Cox/AFT models. Running the parametric models first establishes a performance baseline and validates the survival analysis framing. RSF is then used to explore whether more flexible modelling — at the cost of interpretability — improves customer ranking. Results from all three approaches are compared in a final summary table.

---

## Source Modules

Helper code is organized under `src/sme_kt_zh_collaboration_forecasting/`:

| Module | Purpose |
|---|---|
| `utils.py` | Data loading utility (`read_sales_data`): reads `sales_df.csv`, parses dates, and creates a numeric customer ID while preserving the original name |
| `EDA.py` | Reusable EDA functions: general sales time-series analysis (seasonality plots, MSTL decomposition, stationarity tests), holiday-lag correlation, and AutoGluon training-set builders |
| `modelling.py` | Survival analysis pipeline helpers: data preparation (inter-purchase durations, censoring), train/test splitting, and evaluation utilities (C-index, predicted vs. real priority ranking) for Cox, AFT, and RSF models |

---

## Installation

**Important:** The project uses [`uv`](https://github.com/astral-sh/uv) for dependency management. A `uv.lock` file is included for fully reproducible installs. To use it:
>
> ```bash
> uv sync
> ```


Alternative installation methods follow:

```bash
pip install -r requirements.txt
```

If you are using Conda to manage your Python environments:

```bash
conda env create -f environment.yml
```

Alternatively, if you are using an existing environment, you can install the module in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), which includes only minimal dependencies:

```bash
pip install -e .
```

---

## Development Tools

Register [pre-commit](https://pre-commit.com/) hooks after installation:

```bash
pre-commit install
```

Run hooks manually to verify the setup:

```bash
pre-commit run --all-files
```

Unit tests (via [pytest](https://pytest.org/)) can be run locally:

```bash
pytest
```
