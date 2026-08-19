"""End-to-end pipeline demo for StartupShield AI.

Runs every model, the risk aggregator, and the recommendation engine for all three
demo companies, prints a readable summary, and writes one JSON file per company.

    python run_full_pipeline_demo.py
    python run_full_pipeline_demo.py --output-dir reports/demo
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

from src import demo_data, pipeline


DEFAULT_OUTPUT_DIR = Path("reports/demo")


def _quiet_third_party_logs() -> None:
    """Prophet/cmdstanpy are chatty on import; keep the demo output readable."""
    warnings.filterwarnings("ignore")
    for logger_name in ("cmdstanpy", "prophet", "matplotlib"):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


def summarize(bundle: dict) -> dict:
    """Reduce a scored bundle to the JSON-serializable summary we persist."""
    assessment = bundle["assessment"]
    return {
        "company_name": bundle["company_name"],
        "profile": bundle["profile"],
        "risk_score": round(assessment["risk_score"], 2),
        "risk_label": assessment["risk_label"],
        "signals": {key: round(value, 4) for key, value in assessment["signals"].items()},
        "contributions": {
            key: round(value, 2) for key, value in assessment.get("contributions", {}).items()
        },
        "top_factors": [
            {"feature": name, "shap_value": round(value, 4)}
            for name, value in assessment.get("top_factors", [])
        ],
        "explanation_text": assessment["explanation_text"],
        "recommendations": bundle["recommendations"],
        "data_counts": {
            "customers": len(bundle["customers"]),
            "reviews": len(bundle["reviews"]),
            "daily_metrics": len(bundle["anomalies"]),
            "forecast_rows": len(bundle["forecast"]),
        },
    }


def print_report(summary: dict) -> None:
    """Print one company's summary to the console."""
    print()
    print("=" * 74)
    print(f"  {summary['company_name']}  ({summary['profile']})")
    print("=" * 74)
    print(f"  RISK SCORE : {summary['risk_score']:.1f} / 100   [{summary['risk_label']}]")
    print(f"  WHY        : {summary['explanation_text']}")
    print()
    print("  Signals:")
    for name, value in summary["signals"].items():
        print(f"    - {name:24s} {value:.3f}")
    print()
    print("  Score contributions (points out of 100):")
    for name, value in summary["contributions"].items():
        print(f"    - {name:24s} {value:5.1f}")
    if summary["top_factors"]:
        print()
        print("  Top churn drivers (SHAP):")
        for factor in summary["top_factors"]:
            print(f"    - {factor['feature']:32s} {factor['shap_value']:+.3f}")
    print()
    print("  Recommended actions:")
    for recommendation in summary["recommendations"]:
        print(f"    [{recommendation['priority']:8s}] {recommendation['action']}")
        print(f"               {recommendation['reason']}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the StartupShield AI end-to-end demo.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write per-company summary JSON into.",
    )
    parser.add_argument(
        "--company",
        default="all",
        help="Run a single demo company instead of all of them.",
    )
    return parser.parse_args()


def main() -> None:
    """Score every demo company end to end and persist the summaries."""
    _quiet_third_party_logs()
    args = parse_args()

    companies = (
        demo_data.company_names() if args.company == "all" else [args.company]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trained models...")
    load_started = time.perf_counter()
    models = pipeline.load_models()
    print(f"Models loaded in {time.perf_counter() - load_started:.2f}s")

    summaries = []
    for company_name in companies:
        scoring_started = time.perf_counter()
        bundle = pipeline.score_company(company_name, models)
        elapsed = time.perf_counter() - scoring_started

        summary = summarize(bundle)
        summary["scoring_seconds"] = round(elapsed, 3)
        summaries.append(summary)
        print_report(summary)
        print(f"\n  (scored in {elapsed:.2f}s)")

        slug = "_".join(company_name.lower().split())
        output_path = args.output_dir / f"{slug}.json"
        output_path.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print("=" * 74)
    print("  SUMMARY")
    print("=" * 74)
    for summary in sorted(summaries, key=lambda item: item["risk_score"], reverse=True):
        print(
            f"  {summary['company_name']:22s} {summary['risk_score']:5.1f}  "
            f"{summary['risk_label']:6s}  ({summary['scoring_seconds']:.2f}s)"
        )
    print()
    print(f"Wrote {len(summaries)} summary file(s) to {args.output_dir}/")


if __name__ == "__main__":
    main()
