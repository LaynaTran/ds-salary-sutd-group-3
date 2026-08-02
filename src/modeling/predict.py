"""Predict salary_usd for new rows using the model trained by train.py."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("outputs") / "model.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description="Predict salary_usd for one or more rows.")
    parser.add_argument("--input", type=str, help="Path to a CSV with rows to predict.")
    parser.add_argument("--year", type=int)
    parser.add_argument("--experience_level", type=str, choices=["Entry", "Mid", "Lead", "Senior", "Executive"])
    parser.add_argument("--work_mode", type=str, choices=["Onsite", "Remote", "Hybrid"])
    parser.add_argument("--company_size", type=str, choices=["Small", "Medium", "Large", "Startup", "Enterprise"])
    parser.add_argument("--company_location", type=str)
    parser.add_argument("--general_role", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    model = joblib.load(MODEL_PATH)

    if args.input:
        df = pd.read_csv(args.input)
    else:
        required = ["year", "experience_level", "work_mode", "company_size", "company_location", "general_role"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"Missing required arguments: {', '.join('--' + m for m in missing)}")
        df = pd.DataFrame([{name: getattr(args, name) for name in required}])

    predictions = model.predict(df)
    for i, pred in enumerate(predictions):
        print(f"Row {i}: predicted salary_usd = {pred:,.0f}")


if __name__ == "__main__":
    main()
