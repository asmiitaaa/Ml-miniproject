"""Dashboard integration checks, including interactions with real SUMO evidence.

AppTest opens a local asyncio socketpair. Windows sandbox loopback restrictions
may require running this test command with the same approval as the local server.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
FULL_RESULTS = ROOT / "results/model_validation" if (ROOT / "results/model_validation/runs.csv").is_file() else ROOT / "report/results"
PROVISIONAL_RESULTS = ROOT / "results/provisional" if (ROOT / "results/provisional/runs.csv").is_file() else ROOT / "tests/data/provisional"


def app_for(monkeypatch, path):
    monkeypatch.setattr(sys, "argv", [str(ROOT / "dashboard.py"), "--results", str(path)])
    return AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60).run()


def test_missing_results_gives_actionable_empty_state(monkeypatch, tmp_path):
    app = app_for(monkeypatch, tmp_path / "not_generated")
    assert not app.exception
    assert any("No evaluation data found" in message.value for message in app.info)
    assert any("traffic_eval run" in code.value for code in app.code)
    assert not app.metric


def test_invalid_uploaded_schema_is_not_displayed_as_results(monkeypatch, tmp_path):
    pd.DataFrame({"scenario": ["peak"], "controller": ["dqn"], "seed": [11]}).to_csv(tmp_path / "runs.csv", index=False)
    app = app_for(monkeypatch, tmp_path)
    assert not app.exception
    assert any("Missing columns" in message.value for message in app.error)
    assert not app.metric


@pytest.mark.skipif(not (PROVISIONAL_RESULTS / "runs.csv").is_file(), reason="Real provisional SUMO evidence not generated")
def test_real_runs_filters_recalculate_and_retain_provisional_warning(monkeypatch):
    result_dir = PROVISIONAL_RESULTS
    runs = pd.read_csv(result_dir / "runs.csv")
    app = app_for(monkeypatch, result_dir)
    assert not app.exception
    assert int(app.metric[0].value) == len(runs)
    assert any("not a final project result" in item.value.lower() for item in app.warning)
    assert any("missing controller" in item.value.lower() and "dqn" in item.value.lower() for item in app.warning)
    row = runs.iloc[0]
    app.sidebar.multiselect[0].set_value([row.scenario])
    app.sidebar.multiselect[1].set_value([row.controller])
    app.sidebar.multiselect[2].set_value([int(row.seed)])
    app.run()
    assert not app.exception
    assert [int(item.value) for item in app.metric] == [1, 1, 1, 1]
    assert any("fewer than five" in item.value.lower() for item in app.warning)
    # Exercises a different chart measure and its data path after filtering.
    app.selectbox[0].set_value("co2_kg").run()
    assert not app.exception
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Select at least one scenario" in item.value for item in app.info)


@pytest.mark.skipif(not (FULL_RESULTS / "runs.csv").is_file(), reason="Full DQN validation not generated")
def test_complete_matrix_keeps_dqn_safety_warnings_visible(monkeypatch):
    app=app_for(monkeypatch,FULL_RESULTS)
    assert not app.exception
    assert [int(item.value) for item in app.metric]==[30,2,3,5]
    assert any('collisions' in item.value.lower() for item in app.warning)
    assert not any('missing controller' in item.value.lower() for item in app.warning)
    app.sidebar.multiselect[1].set_value(['dqn']).run()
    assert not app.exception
    assert int(app.metric[0].value)==10


def test_no_traffic_sites_are_visible_even_with_all_thirty_runs(monkeypatch):
    current=ROOT/'report/training/corridor_results'
    app=app_for(monkeypatch,current)
    assert not app.exception
    assert int(app.metric[0].value)==30
    assert any('only 1 of 6' in warning.value for warning in app.warning)
