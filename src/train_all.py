"""Reproducible training entrypoint for all StartupShield AI models.

Rebuilds every model artifact from `data/raw/` so a fresh clone can reproduce the
numbers in `reports/`. Run after `python -m src.generate_synthetic_data`:

    python -m src.train_all
    python -m src.train_all --module churn --seed 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import anomaly_module, churn_module, sentiment_module
from src.generate_synthetic_data import CONFIG_PATH, load_config
from src.preprocessing import train_test_split_stratified


MODULES = ("churn", "sentiment", "anomaly")


def train_churn(config: dict, raw_dir: Path, models_dir: Path) -> dict:
    """Train, compare, and persist the churn candidate models."""
    churn_df = pd.read_csv(raw_dir / "churn.csv")
    features = churn_module.engineer_features(churn_df)
    labels = churn_df[churn_module.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split_stratified(
        features,
        labels,
        test_size=config["churn"]["test_size"],
        seed=config["seed"],
    )

    candidates = churn_module.train_candidate_models(X_train, y_train, seed=config["seed"])
    best_name, best_model, metrics = churn_module.select_best_model(candidates, X_test, y_test)

    churn_module.save_model(best_model, models_dir / "churn_model.pkl")
    for model_name, model in candidates.items():
        churn_module.save_model(model, models_dir / f"churn_model_{model_name}.pkl")

    configured_model = config["churn"]["model_type"]
    if configured_model != best_name:
        print(
            f"  note: config churn.model_type={configured_model!r} but the best "
            f"candidate on this split was {best_name!r}; saved {best_name!r}."
        )

    return {"best_model": best_name, "rows": len(churn_df), "metrics": metrics}


def train_sentiment(config: dict, raw_dir: Path, models_dir: Path) -> dict:
    """Train and persist the TF-IDF sentiment baseline."""
    reviews_df = pd.read_csv(raw_dir / "reviews.csv")
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        reviews_df["review_text"],
        reviews_df["sentiment_label"],
        test_size=0.2,
        seed=config["seed"],
    )

    model = sentiment_module.train_tfidf_baseline(
        X_train,
        y_train,
        max_features=config["sentiment"]["max_features"],
        seed=config["seed"],
    )
    metrics = sentiment_module.evaluate(model, X_test, y_test)
    benchmark = sentiment_module.benchmark_inference(model, list(X_test[:100]))

    sentiment_module.save_model(model, models_dir / "sentiment_model")
    return {
        "best_model": "tfidf_logistic_regression",
        "rows": len(reviews_df),
        "metrics": metrics,
        "inference_seconds_per_100": round(benchmark["seconds"], 4),
    }


def train_anomaly(config: dict, raw_dir: Path, models_dir: Path) -> dict:
    """Train and persist the IsolationForest anomaly detector."""
    timeseries_paths = sorted(raw_dir.glob("timeseries_*.csv"))
    if not timeseries_paths:
        raise FileNotFoundError(f"No timeseries_*.csv files found in {raw_dir}")

    timeseries_df = pd.concat(
        [pd.read_csv(path) for path in timeseries_paths],
        ignore_index=True,
    )
    features = anomaly_module.build_features(timeseries_df)

    model = anomaly_module.train(
        features,
        contamination=config["anomaly"]["contamination"],
        seed=config["seed"],
    )
    metrics = anomaly_module.evaluate_against_ground_truth(
        model,
        features,
        features["is_injected_anomaly"],
    )
    labels, _ = anomaly_module.predict(model, features)

    anomaly_module.save_model(model, models_dir / "anomaly_model.pkl")
    return {
        "best_model": "isolation_forest",
        "rows": len(features),
        "metrics": metrics,
        "detected_anomalies": int((labels == -1).sum()),
        "injected_anomalies": int(features["is_injected_anomaly"].sum()),
    }


def run(modules: tuple[str, ...], config: dict, raw_dir: Path, models_dir: Path) -> dict:
    """Train the requested modules and return their metric summaries."""
    trainers = {
        "churn": train_churn,
        "sentiment": train_sentiment,
        "anomaly": train_anomaly,
    }

    summary = {}
    for module_name in modules:
        print(f"[{module_name}] training...")
        summary[module_name] = trainers[module_name](config, raw_dir, models_dir)
        print(f"[{module_name}] done: {json.dumps(summary[module_name]['metrics'], indent=2)}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train all StartupShield AI models.")
    parser.add_argument(
        "--module",
        choices=[*MODULES, "all"],
        default="all",
        help="Train a single module instead of all of them.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed from config.yaml.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write the metric summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Train models and print a reproducible metric summary."""
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    raw_dir = Path(config["paths"]["raw_data"])
    models_dir = Path(config["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    modules = MODULES if args.module == "all" else (args.module,)
    summary = run(modules, config, raw_dir, models_dir)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote metric summary to {args.summary_json}")

    print("\nAll requested models trained and saved to", models_dir)


if __name__ == "__main__":
    main()
