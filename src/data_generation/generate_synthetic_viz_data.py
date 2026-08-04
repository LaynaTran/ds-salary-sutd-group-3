import itertools

import joblib
import pandas as pd

best_estimator_ = joblib.load("outputs/model.pkl")

# The trained pipeline (best_estimator_) also requires a `work_mode` column
# (one-hot encoded during training as Onsite/Remote/Hybrid). It's fixed to a
# single value here so the requested combination count (8,064) stays exact.
WORK_MODE = "Onsite"

year = [2020, 2021, 2022, 2023, 2024, 2025]
general_role = [
    "Computer Vision Engineer", "Machine Learning Engineer", "Research Engineer",
    "Software Engineer", "Research Scientist", "Solution Architect",
    "Software Architect", "Database Engineer", "Data Scientist", "Data Engineer",
    "Data Analyst", "AI Engineer", "Quantitative Analyst", "Data Visualization",
]
experience_level = ["Entry", "Mid", "Senior", "Lead"]
company_size = ["Small", "Medium", "Large", "Enterprise"]
company_location = [
    "United States", "Australia", "Netherlands", "United Kingdom", "Canada",
    "Italy", "Singapore", "Germany", "Finland",
]

# 1. Generate all feature combinations
combinations = itertools.product(
    year, general_role, experience_level, company_size, company_location
)

# 2. Build the DataFrame
df = pd.DataFrame(
    combinations,
    columns=["year", "general_role", "experience_level", "company_size", "company_location"],
)
df["work_mode"] = WORK_MODE

# 3. Predict and export
df["predicted_salary_usd"] = best_estimator_.predict(df)

df = df.drop(columns=["work_mode"])
export_path = "data/model_generated_data.csv"
df.to_csv(export_path, index=False)

print(f"Total rows generated: {len(df):,}")
print(f"Success: {export_path} exported.")
