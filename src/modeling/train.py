"""Train and compare regression models to predict salary_usd from salary_cleaned.csv."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor

DATA_PATH = Path("data/salary_cleaned.csv")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42

EXPERIENCE_ORDER = ["Entry", "Mid", "Lead", "Senior", "Executive"]
ORDINAL_COLS = ["experience_level"]
ONEHOT_COLS = ["work_mode", "company_size", "company_location", "general_role"]
NUMERIC_COLS = ["year"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["job_title", "company_location_code"])
    df = df.drop_duplicates()
    df = df.dropna(subset=["company_location"])
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("ordinal", OrdinalEncoder(categories=[EXPERIENCE_ORDER]), ORDINAL_COLS),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
            ("numeric", "passthrough", NUMERIC_COLS),
        ]
    )


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_cv(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=3,
        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
        n_jobs=-1,
    )
    return {
        "cv_rmse_mean": float(-scores["test_neg_root_mean_squared_error"].mean()),
        "cv_rmse_std": float(scores["test_neg_root_mean_squared_error"].std()),
        "cv_mae_mean": float(-scores["test_neg_mean_absolute_error"].mean()),
        "cv_r2_mean": float(scores["test_r2"].mean()),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    baseline = None
    metrics_path = OUTPUT_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            baseline = json.load(f)

    print("Loading and cleaning data...")
    df = load_data()
    print(f"  {len(df):,} rows after cleaning")

    X = df.drop(columns=["salary_usd"])
    y = df["salary_usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"  train={len(X_train):,} test={len(X_test):,}")

    # n_jobs=1 on each model: parallelism happens at the cross_validate/RandomizedSearchCV
    # level (n_jobs=-1 there) to avoid oversubscribing cores with nested parallelism.
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
        ),
        "LightGBM": LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
    }

    print("\nComparing candidate models with 3-fold CV on training set...")
    cv_results = {}
    for name, model in candidates.items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        metrics = evaluate_cv(pipeline, X_train, y_train)
        cv_results[name] = metrics
        print(
            f"  {name:15s} RMSE={metrics['cv_rmse_mean']:,.0f} (+/-{metrics['cv_rmse_std']:,.0f})"
            f"  MAE={metrics['cv_mae_mean']:,.0f}  R2={metrics['cv_r2_mean']:.4f}"
        )

    best_name = min(cv_results, key=lambda n: cv_results[n]["cv_rmse_mean"])
    print(f"\nBest candidate by CV RMSE: {best_name}")

    param_distributions = {
        "RandomForest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 20, 30],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        "XGBoost": {
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [4, 6, 8, 10],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
        },
        "LightGBM": {
            "model__n_estimators": [200, 300, 500, 800],
            "model__num_leaves": [15, 31, 63, 127],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
        },
        "LinearRegression": {},
    }

    best_pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", candidates[best_name])])
    param_grid = param_distributions.get(best_name, {})

    if param_grid:
        print(f"\nTuning {best_name} with RandomizedSearchCV...")
        search = RandomizedSearchCV(
            best_pipeline,
            param_distributions=param_grid,
            n_iter=20,
            scoring="neg_root_mean_squared_error",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        final_model = search.best_estimator_
        print(f"  Best params: {search.best_params_}")
        print(f"  Best CV RMSE: {-search.best_score_:,.0f}")
    else:
        final_model = best_pipeline.fit(X_train, y_train)

    print("\nEvaluating on held-out test set...")
    y_pred = final_model.predict(X_test)
    test_metrics = {
        "rmse": rmse(y_test, y_pred),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    print(
        f"  Test RMSE={test_metrics['rmse']:,.0f}  MAE={test_metrics['mae']:,.0f}  R2={test_metrics['r2']:.4f}"
    )

    # Residuals plot
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.2, s=8)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("Actual salary_usd")
    axes[0].set_ylabel("Predicted salary_usd")
    axes[0].set_title("Predicted vs Actual")

    axes[1].hist(residuals, bins=60)
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "residuals.png", dpi=150)
    plt.close(fig)

    # Feature importance plot (tree-based models only)
    model_step = final_model.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        feature_names = final_model.named_steps["preprocess"].get_feature_names_out()
        importances = model_step.feature_importances_
        order = np.argsort(importances)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(range(len(order)), importances[order][::-1])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 20 feature importances ({best_name})")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)

    joblib.dump(final_model, OUTPUT_DIR / "model.pkl")

    report = {
        "best_model": best_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    if baseline:
        print("\nComparison vs previous run (one-hot encoding, raw-dollar target):")
        for name, metrics in cv_results.items():
            old = baseline.get("cv_results", {}).get(name)
            if old:
                print(
                    f"  {name:15s} RMSE {old['cv_rmse_mean']:,.0f} -> {metrics['cv_rmse_mean']:,.0f}"
                    f"   R2 {old['cv_r2_mean']:.4f} -> {metrics['cv_r2_mean']:.4f}"
                )
        old_test = baseline.get("test_metrics")
        if old_test:
            print(
                f"  {'Final test':15s} RMSE {old_test['rmse']:,.0f} -> {test_metrics['rmse']:,.0f}"
                f"   R2 {old_test['r2']:.4f} -> {test_metrics['r2']:.4f}"
            )

    print(f"\nSaved model.pkl, metrics.json, and plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
