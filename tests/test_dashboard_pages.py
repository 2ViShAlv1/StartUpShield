"""Smoke tests for the Streamlit dashboard pages.

These guard the demo: every page must render for every demo company without
raising, because a crash during a live demo is unrecoverable.
"""

import logging
import warnings

import pytest

from src import demo_data

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


PAGE_PATHS = [
    "app/app.py",
    "app/pages/0_Upload_Your_Company.py",
    "app/pages/1_Churn.py",
    "app/pages/2_Sentiment.py",
    "app/pages/3_Anomalies.py",
    "app/pages/4_Forecast.py",
    "app/pages/5_Risk_and_Recommendations.py",
]


@pytest.fixture(autouse=True)
def _quiet_third_party_logs():
    """Prophet and matplotlib are noisy on import; keep test output readable."""
    warnings.filterwarnings("ignore")
    for logger_name in ("cmdstanpy", "prophet", "matplotlib"):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


@pytest.mark.parametrize("page_path", PAGE_PATHS)
@pytest.mark.parametrize("company_name", demo_data.company_names())
def test_page_renders_without_exception(page_path: str, company_name: str) -> None:
    """Every dashboard page must render for every demo company."""
    app = AppTest.from_file(page_path, default_timeout=180)
    app.session_state["company_name"] = company_name
    app.run()

    assert not app.exception, (
        f"{page_path} raised for {company_name}: "
        f"{app.exception[0].value if app.exception else ''}"
    )


def test_overview_shows_a_risk_score_for_each_company() -> None:
    """The headline number is the whole product -- it must always be present."""
    for company_name in demo_data.company_names():
        app = AppTest.from_file("app/app.py", default_timeout=180)
        app.session_state["company_name"] = company_name
        app.run()

        assert not app.exception
        metric_labels = [metric.label for metric in app.metric]
        assert "Mean churn probability" in metric_labels
        assert any(company_name in header.value for header in app.subheader)

        badge_markdown = " ".join(block.value or "" for block in app.markdown)
        assert "/ 100" in badge_markdown
        assert "Risk" in badge_markdown
