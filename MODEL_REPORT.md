# Salary Prediction Model — Project Report

**Objective:** Predict `salary_usd` for tech/data industry roles using structured attributes (role, experience level, location, company size, work mode, and year), based on `data/salary_cleaned.csv` (110,526 raw records; 99,425 after cleaning).

**Status:** Training and optimization complete. The final model is production-ready *for the use cases described in [Section 4](#4-practical-applications--how-to-use-this-model)*, and not for others.

This repository has two components:
- **Main part** — the full model development process (Sections 1–4 below), implemented in [`src/modeling/`](src/modeling/): data cleaning, model comparison, hyperparameter tuning, ablation experiments, and the final evaluated model.
- **Secondary part** — a supplementary script in [`src/data_generation/`](src/data_generation/) that uses the final trained model to generate a synthetic dataset for downstream visualization (see [Section 5](#5-secondary-component--synthetic-data-generation-for-visualization)).

---

## 1. Project Summary & Final Results

We benchmarked four model families under identical preprocessing and a consistent 80/20 train/test split, using 3-fold cross-validation for model selection and `RandomizedSearchCV` for tuning the winner.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | $56,416 | $39,016 | 0.393 |
| Random Forest | $54,146 | $34,770 | 0.441 |
| **XGBoost (tuned, final)** | **$53,800** (test) | **$34,348** | **0.453** |
| LightGBM | $53,703 | $34,289 | 0.450 |

> **Bottom line:** XGBoost and LightGBM are effectively tied for best performance, with XGBoost selected as the final model after hyperparameter tuning. All three tree-based models comfortably outperform the linear baseline, confirming that salary is driven by *non-linear* interactions between role, location, and experience — not a simple additive formula.

The final model artifact (`outputs/model.pkl`) bundles preprocessing and the tuned XGBoost regressor into a single deployable pipeline.

---

## 2. Methodology & Technical Decisions (The "Why")

Our workflow followed a standard, defensible progression — but with one honest detour worth reporting in full, because *what didn't work* is as informative as what did.

### 2.1 Baselines → Tuning → Feature Engineering
1. **Baselines:** We first trained four candidates — Linear Regression, Random Forest, XGBoost, LightGBM — with sensible default hyperparameters, evaluated via 3-fold cross-validation on RMSE, MAE, and R².
2. **Hyperparameter tuning:** The best baseline candidate (XGBoost) was tuned using `RandomizedSearchCV` across tree depth, learning rate, subsampling, and column sampling, then re-evaluated on a held-out test set never touched during tuning.
3. **Feature engineering experiments:** With a working baseline in hand, we tested two well-established techniques specifically aimed at *improving* on it.

### 2.2 The Ablation Test — A Plot Twist Worth Reporting

We hypothesized that two changes would improve model quality:

- **Target Encoding** (with proper K-fold / out-of-fold cross-fitting to prevent leakage) for the high-cardinality categorical features `company_location` (96 countries) and `general_role` (36 categories), in place of one-hot encoding.
- **Log-transformation** of `salary_usd` (via a `TransformedTargetRegressor`), intended to reduce the influence of high-earner outliers and correct for the right-skew in the target distribution.

Both are textbook techniques for exactly this kind of data. **Both backfired.**

| Configuration | RMSE | R² |
|---|---|---|
| One-hot + raw-dollar target *(baseline)* | $54,146 | 0.441 |
| One-hot + **log** target | $54,851 | 0.426 |
| **Target-encoded** + raw-dollar target | $56,707 | 0.387 |
| Target-encoded + log target *(combined)* | $56,539 | 0.390 |

*(Random Forest shown as the representative model for this ablation; the same direction of effect held across all four candidates.)*

**Why this matters:** it's tempting to assume "more sophisticated" preprocessing is strictly better. It isn't — it's an empirical question, and here the empirical answer was "no." Target encoding collapses each category into a single averaged number, which strips the tree-based models of the ability to make clean, independent splits on individual categories (e.g., isolating "United States" or "Engineering Manager" as its own binary decision). One-hot encoding preserves that flexibility, and — despite being the "simpler" method — proved genuinely more informative for this dataset's tree ensembles.

### 2.3 Final Decision

> We deliberately **reverted to one-hot encoding and a raw-dollar target**, not because it was the default or the easy path, but because it *empirically outperformed* the more sophisticated alternatives on this dataset. **XGBoost** was selected as the final production model (LightGBM as a close, viable second), both tuned and validated on held-out data.

This is intellectually honest data science: hypotheses were tested, evidence was collected, and the simpler approach won on the numbers — not on convention.

---

## 3. Data Limitations & Industry Standards (Managing Expectations)

### 3.1 What the metrics actually mean

- **R² = 0.453** — the model explains roughly **45% of the variance** in salaries using role, experience, location, company size, work mode, and year. The remaining **~55% is unexplained** by these features.
- **MAE ≈ $34k vs. RMSE ≈ $54k** — this gap is diagnostic, not incidental. RMSE penalizes large errors disproportionately (it squares them before averaging), while MAE treats all errors equally. The sizeable gap between them tells us the *typical* prediction is only off by roughly $34k, but a *subset* of predictions — concentrated among very high earners — are off by much larger amounts, and those outliers are dragging RMSE upward. Our residual analysis confirms this: the model systematically under-predicts salaries above ~$300k.

### 3.2 Where the other ~55% is hiding

The dataset captures *structural* factors — role, seniority, geography, company size. It cannot capture the *individual and organizational* factors that meaningfully move compensation in the real world, including:

- Individual negotiation skill and interview performance
- Specific company budget, funding stage, and profitability
- Equity, stock options, signing bonuses, and other non-salary compensation
- Niche or in-demand technical expertise not captured by job title alone
- Internal pay bands, manager discretion, and counter-offer dynamics

No dataset of this shape — role/location/experience-level records without company identifiers or individual performance data — could close this gap. This is a *ceiling of the data*, not a shortcoming of the modeling.

### 3.3 Is 0.45 actually a *good* R²?

> **Yes.** For human-centric, behavioral outcomes like compensation, an R² in the **0.4–0.5** range is a normal, credible result — not a disappointing one.

Compensation is the outcome of negotiation, individual variation, and organizational discretion, not a deterministic function of a candidate's résumé attributes. Counterintuitively, **an R² of 0.8–0.9 on this kind of data would be a red flag for data leakage** (for example, a feature that indirectly encodes the target, such as a company identifier correlated with a narrow pay band). We have reached the **information ceiling** of this dataset with the current feature set — further gains would require *new data* (e.g., company-level identifiers, compensation components beyond base salary), not further modeling effort on the existing columns.

---

## 4. Practical Applications — How to Use This Model

The error margins established above (± $34k typical, larger for high earners) directly determine what this model is, and is not, fit for.

### ✅ Appropriate Use Cases

- **Macro market trend analysis** — tracking how average compensation shifts across years, regions, or role categories over time.
- **Compensation disparity analysis** — comparing typical pay across geographies, company sizes, or work modes to identify structural gaps (e.g., quantifying the remote-vs-onsite pay differential seen in this data).
- **Reference salary banding** — providing a general, *directional* estimate for planning purposes, always presented as a range: *"the market rate for this profile is approximately $100k, ± $34k."*

### 🚫 Inappropriate Use Cases

> **This model must not be used to automatically set, lock in, or dictate a specific salary offer for an individual candidate.**

- It cannot see the ~55% of factors — negotiation, company budget, equity, specific expertise, interview performance — that legitimately influence an individual offer.
- Automating an offer from this model's point estimate would mean systematically ignoring exactly the information a human recruiter or hiring manager is best positioned to weigh.
- **Recommendation:** use this model to inform and calibrate the *starting point* of a compensation conversation, never to replace the conversation itself.

---

## 5. Secondary Component — Synthetic Data Generation for Visualization

[`src/data_generation/generate_synthetic_viz_data.py`](src/data_generation/generate_synthetic_viz_data.py) is a supplementary script, separate from the core evaluation above. It uses the final trained model purely as a **scoring engine** to produce a dataset shaped for charting rather than for model assessment:

1. Builds the full factorial grid (`itertools.product`) of `year` (2020–2023) × `general_role` (14 roles) × `experience_level` (4 levels) × `company_size` (4 sizes) × `company_location` (9 countries) — **8,064 combinations**, each an input profile that may or may not exist in the real training data.
2. `work_mode` is held constant (`"Onsite"`) since the model requires it but it isn't part of the requested grid dimensions.
3. Feeds every combination through `outputs/model.pkl` to get `predicted_salary_usd`, and exports the result to `outputs/synthetic_salary_data_for_viz.csv`.

This is exploratory tooling for building charts (e.g. predicted pay by role × country, or by experience level over time) — it does **not** evaluate the model and carries no accuracy claims beyond what Sections 1–3 already established.

---