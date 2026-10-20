"""Checks for the deployment features: FHIR export, family message drafts, live replay. Run: pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heatshield import advisory, fhir                            # noqa: E402
from heatshield.config import PROCESSED, steps_per_day, cfg      # noqa: E402

ROW = pd.Series({"name": "Suresh Yadav", "sex": "M", "city": "Nagpur", "outdoor_routine": 1, "home_ac": 0})
T0 = pd.Timestamp("2024-05-27 20:00")


def test_fhir_bundle_is_valid_shape():
    ra = fhir.risk_assessment("P041", T0, T0 + pd.Timedelta("5min"), T0 + pd.Timedelta("24h"), 0.93, 9.6,
                              (pd.Timestamp("2024-05-28 11:50"), pd.Timestamp("2024-05-28 15:40")), ["heat"], "stay in")
    b = fhir.bundle([fhir.patient_resource(ROW, "P041"), ra, fhir.core_temp_observation("P041", T0, 37.4, 0.1)])
    b = json.loads(fhir.dumps(b))                                  # serialisable
    assert b["resourceType"] == "Bundle" and len(b["entry"]) == 3
    ra = b["entry"][1]["resource"]
    assert ra["resourceType"] == "RiskAssessment"
    assert ra["subject"]["reference"] == "Patient/P041"
    pred = ra["prediction"][0]
    assert 0 <= pred["probabilityDecimal"] <= 1
    assert pred["qualitativeRisk"]["coding"][0]["code"] == "high"
    assert pred["whenPeriod"]["start"] < pred["whenPeriod"]["end"]
    assert "SYNTHETIC" in ra["note"][0]["text"]
    obs = b["entry"][2]["resource"]
    assert obs["code"]["coding"][0]["code"] == "8310-5" and obs["valueQuantity"]["code"] == "Cel"


def test_family_message_is_safe():
    m = advisory.draft(ROW, 0.9, (pd.Timestamp("2024-05-28 11:50"), pd.Timestamp("2024-05-28 15:40")), T0)
    assert m["level"] == "high"
    assert "11:50 AM" in m["en"] and "3:40 PM" in m["en"]
    assert "108" in m["en"] and "108" in m["hi"]
    # never tells the family to stop a medicine; always defers to the doctor
    assert "Do not stop or change any medicine" in m["en"]
    assert "दवा बंद या कम न करें" in m["hi"]
    assert advisory.draft(ROW, 0.05, None, T0)["level"] == "low"


def test_live_replay_runs_and_alert_is_consistent():
    if not (PROCESSED / "world.npz").exists():
        pytest.skip("run experiments/run_all.py first")
    from heatshield.twin.twin import PatientTwin
    cohort = pd.read_parquet(PROCESSED / "cohort.parquet")
    D = dict(np.load(PROCESSED / "world.npz"))
    t0 = 5 * steps_per_day() + cfg()["outcome"]["evening_hour"] * 12
    rp = PatientTwin(cohort, D, cohort["pid"].iloc[0]).replay(t0, hours=6, every_h=2)
    assert len(rp["frames"]) == 3
    assert all(0 <= f["p_event"] <= 1 for f in rp["frames"])
    if rp["alert"] is not None:
        assert rp["alert"]["p_event"] >= rp["alert_p"]
    assert rp["obs"]["tc_est"].between(35, 42).all()
