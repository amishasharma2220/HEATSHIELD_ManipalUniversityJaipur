"""E5: the what-if engine must behave sensibly. Run: pytest -q

Each check runs on a small set of real cohort patients from the evening of day 6.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heatshield.config import PROCESSED, steps_per_day, cfg      # noqa: E402
from heatshield.twin import filter as F                          # noqa: E402
from heatshield.twin.scenarios import Scenario, build_horizon    # noqa: E402
from heatshield.psi import psi                                   # noqa: E402

N_PATIENTS = 24


@pytest.fixture(scope="module")
def setup():
    if not (PROCESSED / "world.npz").exists():
        pytest.skip("run experiments/run_all.py first")
    cohort = pd.read_parquet(PROCESSED / "cohort.parquet")
    D = dict(np.load(PROCESSED / "world.npz"))
    n = min(N_PATIENTS, len(D["hr_obs"]))
    cohort = cohort.iloc[:n].reset_index(drop=True)
    D = {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(D["hr_obs"]) else v)
         for k, v in D.items()}
    t0 = 5 * steps_per_day() + cfg()["outcome"]["evening_hour"] * 12
    post = F.run_filter(cohort, D, "twin", snapshot_steps={t0}, t_end=t0 + 1, N=200)
    return cohort, D, t0, post["snapshots"][t0]


def run(setup, sc, cohort=None):
    c, D, t0, cloud = setup
    c = c if cohort is None else cohort
    H, k, _ = build_horizon(c, D, t0, steps_per_day(), sc)
    return F.forecast(cloud, k, H, seed=11)       # same seed: differences come from the scenario only


TOL = 1e-6


def test_hotter_day_never_lowers_strain(setup):
    base, hot = run(setup, Scenario()), run(setup, Scenario(temp_offset_c=3))
    assert np.all(hot["peak_psi"] >= base["peak_psi"] - TOL)
    assert np.all(hot["peak_tc"] >= base["peak_tc"] - TOL)


def test_cooler_day_never_raises_strain(setup):
    base, cool = run(setup, Scenario()), run(setup, Scenario(temp_offset_c=-3))
    assert np.all(cool["peak_tc"] <= base["peak_tc"] + TOL)


def test_staying_indoors_never_raises_strain(setup):
    base, ind = run(setup, Scenario()), run(setup, Scenario(indoors_from_h=10, indoors_to_h=18))
    assert np.all(ind["peak_tc"] <= base["peak_tc"] + TOL)


def test_lighter_activity_never_raises_strain(setup):
    base, light = run(setup, Scenario()), run(setup, Scenario(activity_scale=0.6))
    assert np.all(light["peak_tc"] <= base["peak_tc"] + TOL)


def test_extra_fluids_never_raise_strain(setup):
    base, fl = run(setup, Scenario()), run(setup, Scenario(extra_fluids=True))
    assert np.all(fl["peak_tc"] <= base["peak_tc"] + TOL)


def test_power_cut_only_matters_with_ac(setup):
    c = setup[0]
    base, cut = run(setup, Scenario()), run(setup, Scenario(power_cut=True))
    ac = c["home_ac"].to_numpy() == 1
    assert np.all(cut["peak_tc"] >= base["peak_tc"] - TOL)
    np.testing.assert_allclose(cut["peak_tc"][~ac], base["peak_tc"][~ac])


def test_removing_diuretic_only_affects_diuretic_patients(setup):
    c = setup[0]
    base, nod = run(setup, Scenario()), run(setup, Scenario(no_diuretic=True))
    diu = c["diuretic"].to_numpy() == 1
    np.testing.assert_allclose(nod["peak_tc"][~diu], base["peak_tc"][~diu])
    assert np.all(nod["peak_tc"][diu] <= base["peak_tc"][diu] + TOL)
    assert Scenario(no_diuretic=True).hypothetical


def test_probabilities_are_valid(setup):
    r = run(setup, Scenario())
    assert np.all((r["p_event"] >= 0) & (r["p_event"] <= 1))
    assert np.all(r["psi_lo"] <= r["psi_hi"] + TOL)


def test_psi_formula_endpoints():
    assert psi(39.5, 180, 37.0, 70) == pytest.approx(10.0)
    assert psi(37.0, 70, 37.0, 70) == pytest.approx(0.0)
    assert psi(38.25, 125, 37.0, 70) == pytest.approx(5.0)
