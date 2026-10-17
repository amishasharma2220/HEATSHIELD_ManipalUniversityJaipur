"""One row per (patient, evening): features known at 20:00 and the label for the next 24 h."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import cfg, steps_per_day
from ..psi import is_event
from ..weather import heat_index_c

EHR_COLS = ["age", "egfr", "weight_kg", "beta_blocker", "diuretic", "sglt2i", "acei_arb", "t2d", "ckd",
            "home_ac", "outdoor_routine", "morning_walk"]
WEAR_COLS = ["hr_mean_24h", "hr_p95_24h", "hr_above_night", "skin_max_24h", "steps_24h", "sleep_h_24h"]
WX_COLS = ["fc_tmax", "fc_heat_index_max", "fc_hours_above_40", "fc_night_min"]
TWIN_COLS = ["twin_tc_now", "twin_deficit_now", "twin_hr_gain", "twin_hr_rest", "twin_p_event",
             "twin_peak_psi", "twin_peak_tc"]


def evening_steps() -> list[int]:
    spd = steps_per_day()
    eh = cfg()["outcome"]["evening_hour"] * 60 // cfg()["simulation"]["step_minutes"]
    days = cfg()["simulation"]["days"]
    return [d * spd + eh for d in range(days) if d * spd + eh + spd < days * spd]


def build(cohort: pd.DataFrame, D: dict, twin_fc: dict) -> pd.DataFrame:
    """twin_fc[t0] = forecast() output dict for all patients from that evening."""
    spd = steps_per_day()
    step_h = cfg()["simulation"]["step_minutes"] / 60
    rows = []
    for k, t0 in enumerate(evening_steps()):
        past = slice(max(t0 - spd + 1, 0), t0 + 1)
        nxt = slice(t0 + 1, t0 + 1 + spd)
        hr = D["hr_obs"][:, past]
        night = np.nanmedian(D["hr_obs"][:, max(t0 - spd + 1, 0):t0 + 1][:, -(20 * 12):-(14 * 12)], axis=1)
        f = pd.DataFrame({"pid": cohort["pid"], "evening": k + 1, "t0": t0})
        for c in EHR_COLS:
            f[c] = cohort[c].to_numpy()
        f["hr_mean_24h"] = np.nanmean(hr, 1)
        f["hr_p95_24h"] = np.nanpercentile(hr, 95, axis=1)
        f["hr_above_night"] = f["hr_p95_24h"] - night
        f["skin_max_24h"] = np.nanmax(D["skin_obs"][:, past], 1)
        f["steps_24h"] = D["steps"][:, past].sum(1)
        f["sleep_h_24h"] = D["sleep"][:, past].sum(1) * step_h
        fc_t = D["t_out_forecast"][:, nxt]
        f["fc_tmax"] = fc_t.max(1)
        f["fc_heat_index_max"] = heat_index_c(fc_t, D["rh"][:, nxt]).max(1)
        f["fc_hours_above_40"] = (fc_t > 40).sum(1) * step_h
        f["fc_night_min"] = fc_t.min(1)
        tf = twin_fc[t0]
        f["twin_tc_now"], f["twin_deficit_now"] = tf["tc_now"], tf["def_now"]
        f["twin_hr_gain"], f["twin_hr_rest"] = tf["hr_gain"], tf["hr_rest"]
        f["twin_p_event"], f["twin_peak_psi"], f["twin_peak_tc"] = tf["p_event"], tf["peak_psi"], tf["peak_tc"]
        # label from HIDDEN ground truth
        f["label"] = is_event(D["true_psi"][:, nxt].max(1), D["true_tc"][:, nxt].max(1)).astype(int)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)
