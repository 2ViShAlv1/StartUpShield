"""Smoke tests for the Streamlit dashboard pages.

These guard the demo: every page must render for every demo company without
raising, because a crash during a live demo is unrecoverable.
"""

import logging
import warnings
from pathlib import Path

import pytest

from src import demo_data

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

# Streamlit >=1.6x resolves AppTest.from_file's relative paths against the calling
# file's own directory, not the pytest rootdir -- a bare "app/app.py" here would
# resolve to tests/app/app.py and raise FileNotFoundError. Anchoring to the repo
# root keeps this working across streamlit versions.
REPO_ROOT = Path(__file__).resolve().parent.parent

PAGE_PATHS = [
    str(REPO_ROOT / "app" / "app.py"),
    str(REPO_ROOT / "app" / "pages" / "0_Upload_Your_Company.py"),
    str(REPO_ROOT / "app" / "pages" / "1_Churn.py"),
    str(REPO_ROOT / "app" / "pages" / "2_Sentiment.py"),
    str(REPO_ROOT / "app" / "pages" / "3_Anomalies.py"),
    str(REPO_ROOT / "app" / "pages" / "4_Forecast.py"),
    str(REPO_ROOT / "app" / "pages" / "5_Risk_and_Recommendations.py"),
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
        app = AppTest.from_file(str(REPO_ROOT / "app" / "app.py"), default_timeout=180)
        app.session_state["company_name"] = company_name
        app.run()

        assert not app.exception
        metric_labels = [metric.label for metric in app.metric]
        assert "Mean churn probability" in metric_labels
        assert any(company_name in header.value for header in app.subheader)

        badge_markdown = " ".join(block.value or "" for block in app.markdown)
        assert "/ 100" in badge_markdown
        assert "Risk" in badge_markdown
