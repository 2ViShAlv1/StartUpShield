"""Shared helpers for the StartupShield AI Streamlit dashboard.

Keeps model loading and per-company scoring in one cached place so every page
reuses the same work instead of recomputing it on each navigation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import demo_data, pipeline  # noqa: E402
from src.generate_synthetic_data import CONFIG_PATH, load_config  # noqa: E402


RISK_COLOURS = {"green": "#1a7a3c", "orange": "#d9821f", "red": "#c0392b"}
SENTIMENT_COLOURS = {"positive": "#1a7a3c", "neutral": "#8a94a0", "negative": "#c0392b"}


@st.cache_resource(show_spinner="Preparing models (first run may take a few seconds)...")
def get_models() -> dict:
    """Load every trained model, bootstrapping them on first run if needed.

    Deployment matters here. `models/*.pkl` is gitignored, so a fresh deploy (or a
    fresh clone) has no model artifacts at all. Rather than committing ~11 MB of
    pickles -- which are fragile across library versions and would silently fail to
    unpickle on a host with a different sklearn -- the models are trained in the
    deployment environment on first load. A full train takes a few seconds, and
    `@st.cache_resource` means it happens once per container, not per user.

    The raw CSVs in `data/raw/` ARE committed: they are small, plain text, and carry
    no version fragility, so they act as the reproducible seed for this.
    """
    models_dir = REPO_ROOT / "models"
    models = pipeline.load_models(models_dir)
    if _models_are_complete(models):
        return models

    _bootstrap_models(models_dir)
    return pipeline.load_models(models_dir)


def _models_are_complete(models: dict) -> bool:
    """True when every model the dashboard needs actually loaded."""
    if not all(models.get(name) is not None for name in ("churn", "sentiment", "anomaly")):
        return False
    forecast_models = models.get("forecast", {})
    return bool(forecast_models) and all(model is not None for model in forecast_models.values())


def _bootstrap_models(models_dir: Path) -> None:
    """Generate raw data if absent, then train every model into `models_dir`."""
    from src import train_all
    from src.generate_synthetic_data import write_phase1_datasets

    config = get_config()
    raw_dir = REPO_ROOT / config["paths"]["raw_data"]

    required_files = ["churn.csv", "reviews.csv"]
    has_timeseries = any(raw_dir.glob("timeseries_*.csv")) if raw_dir.exists() else False
    if not raw_dir.exists() or not has_timeseries or any(
        not (raw_dir / name).exists() for name in required_files
    ):
        raw_dir.mkdir(parents=True, exist_ok=True)
        write_phase1_datasets(raw_dir, seed=config.get("seed", 42))

    models_dir.mkdir(parents=True, exist_ok=True)
    train_all.run(train_all.MODULES, config, raw_dir, models_dir)


@st.cache_resource
def get_config() -> dict:
    """Load project configuration once per server session."""
    return load_config(REPO_ROOT / CONFIG_PATH)


@st.cache_data(show_spinner="Scoring company...")
def get_company_bundle(company_name: str) -> dict:
    """Run the full pipeline for one company, cached per company."""
    return pipeline.score_company(company_name, get_models(), get_config())


def page_setup(title: str, icon: str = "🛡️") -> None:
    """Apply the shared page config and header styling."""
    st.set_page_config(page_title=f"StartupShield AI — {title}", page_icon=icon, layout="wide")


def company_selector() -> str:
    """Render the sidebar company picker, persisting the choice across pages."""
    names = demo_data.company_names()
    if "company_name" not in st.session_state:
        st.session_state.company_name = names[0]

    selected = st.sidebar.selectbox(
        "Company",
        names,
        index=names.index(st.session_state.company_name),
        key="company_selector",
    )
    st.session_state.company_name = selected

    st.sidebar.caption(f"Profile: **{demo_data.company_profile(selected)}**")
    st.sidebar.divider()
    st.sidebar.caption(
        "Demo data notice: customer and review portfolios are curated demo slices, "
        "and the review/time-series datasets are synthetic. See README."
    )
    return selected


def risk_badge(assessment: dict) -> None:
    """Render the headline risk score as a large coloured badge."""
    colour = RISK_COLOURS.get(assessment["risk_colour"], "#444")
    st.markdown(
        f"""
        <div style="background:{colour};color:#fff;padding:18px 24px;border-radius:12px;
                    display:flex;align-items:baseline;gap:18px;">
          <span style="font-size:52px;font-weight:700;line-height:1;">
            {assessment['risk_score']:.0f}
          </span>
          <span style="font-size:16px;opacity:.85;">/ 100</span>
          <span style="font-size:26px;font-weight:600;margin-left:auto;">
            {assessment['risk_label']} Risk
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str) -> None:
    """Consistent placeholder when a company has no data for a page."""
    st.info(f"No data available — {message}")
