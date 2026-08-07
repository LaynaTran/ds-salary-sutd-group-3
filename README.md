# Data Science Salary prediction model & Visualization

## Project structure

```
SUTD proj v2/
├── MODEL_REPORT.md
├── README.md
├── requirements.txt
├── EDA_with_generated.ipynb          # Exploratory data analysis & visualization
├── data/
│   ├── salary_cleaned.csv            # Training data
│   └── model_generated_data.csv      # Data generated using the final model, for visualization
├── src/
│   ├── modeling/
│   │   ├── train.py                  # Trains/compares/tunes models
│   │   └── predict.py                # CLI to predict from the saved model
│   └── data_generation/
│       └── generate_synthetic_viz_data.py  # Generates salary data using the final model, for visualization
└── outputs/
    ├── model.pkl                     # Final trained model
    ├── metrics.json                  # CV + test metrics for all candidate models
    ├── feature_importance.png
    └── residuals.png
```

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

A trained `outputs/model.pkl` is already included, so this step is only needed if you want to retrain from scratch. Writes to `outputs/model.pkl`, `metrics.json`, `feature_importance.png`, and `residuals.png`.

Re-runs the full pipeline — cleaning, model comparison, tuning, evaluation — and overwrites everything in `outputs/`:

```bash
python src/modeling/train.py
```

### 3. Predict

Predict single profile:

```bash
python src/modeling/predict.py \
    --year 2025 \
    --experience_level Senior \
    --work_mode Remote \
    --company_size Medium \
    --company_location "United States" \
    --general_role "Data Scientist"
```

Valid values: `experience_level` ∈ {Entry, Mid, Lead, Senior, Executive}; `work_mode` ∈ {Onsite, Remote, Hybrid}; `company_size` ∈ {Small, Medium, Large, Startup, Enterprise}; `company_location` is any country name as it appears in the source data; `general_role` is any of the 36 role categories (e.g. "Data Scientist", "Machine Learning Engineer", "Engineering Manager")

Predict in batch. Provide a CSV with columns `year, experience_level, work_mode, company_size, company_location, general_role`:

```bash
python src/modeling/predict.py --input rows.csv
```

### 4. Generate data for visualization

Uses the final trained model to score a grid of profiles and writes the result to `data/model_generated_data.csv`.

```bash
python src/data_generation/generate_synthetic_viz_data.py
```

### 5. Explore the notebook

Notebook downloads datasource, clean it, output cleaned data, then uses `data/model_generated_data.csv` to generate final visualizations.

```bash
jupyter lab EDA_with_generated.ipynb
```
