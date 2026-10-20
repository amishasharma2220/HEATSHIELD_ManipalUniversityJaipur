"""PatientTwin: one living virtual patient, used by the dashboard.

update(t)   -> filter the wearable stream up to step t (the twin 'catches up' with the patient)
forecast()  -> next 24 h under any scenario, with uncertainty
explain()   -> which factors drive tomorrow's risk
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd

from ..config import cfg, steps_per_day
from ..psi import psi
from . import filter as F
from .model import PARAMS, TC_REST
from .scenarios import Scenario, build_horizon


class PatientTwin:
    def __init__(self, cohort: pd.DataFrame, D: dict, pid: str, seed: int = 0):
        self.i = int(np.where(cohort["pid"].to_numpy() == pid)[0][0])
        self.profile = cohort.iloc[[self.i]].reset_index(drop=True)
        self.D = {k: (v[[self.i]] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in D.items()}
        self.seed = seed
        self.t = None
        self.post = None
        self.cloud = None

    # ------------------------------------------------------------------ update
    def update(self, t: int):
        """Run the filter through every observation up to and including step t."""
        self.post = F.run_filter(self.profile, self.D, "twin", snapshot_steps=(t,), t_end=t + 1, seed=self.seed)
        self.cloud = self.post["snapshots"][t]
        self.t = t
        return self

    def state(self) -> dict:
        p = self.post
        t = self.t
        return {"tc": p["tc_mean"][0, t], "tc_sd": p["tc_sd"][0, t], "deficit_l": p["def_mean"][0, t],
                **{k: p["params"][k][0, t] for k in PARAMS}, **{f"{k}_sd": p["params_sd"][k][0, t] for k in PARAMS}}

    def history(self) -> pd.DataFrame:
        """Observed + estimated series up to now (for the live-state tab)."""
        t = self.t + 1
        step = cfg()["simulation"]["step_minutes"]
        start = pd.Timestamp(str(self.D["start"][self.i])) if self.D["start"].ndim else pd.Timestamp(str(self.D["start"]))
        df = pd.DataFrame({
            "time": start + pd.to_timedelta(np.arange(t) * step, unit="min"),
            "hr_obs": self.D["hr_obs"][0, :t], "skin_obs": self.D["skin_obs"][0, :t],
            "steps": self.D["steps"][0, :t], "t_out": self.D["t_out"][0, :t], "t_in": self.D["t_in"][0, :t],
            "outdoor": self.D["outdoor"][0, :t], "tc_est": self.post["tc_mean"][0, :t],
            "tc_sd": self.post["tc_sd"][0, :t], "deficit_est": self.post["def_mean"][0, :t],
        })
        for k in PARAMS:
            df[k] = self.post["params"][k][0, :t]
            df[k + "_sd"] = self.post["params_sd"][k][0, :t]
        return df

    # ------------------------------------------------------------------ forecast
    def forecast(self, scenario: Scenario | None = None, hours: int = 24, seed: int = 1) -> dict:
        h = hours * 60 // cfg()["simulation"]["step_minutes"]
        h = min(h, self.D["t_out"].shape[1] - self.t - 1)
        H, k, hrs = build_horizon(self.profile, self.D, self.t, h, scenario)
        res = F.forecast(self.cloud, k, H, seed=seed, spread=F.unlearned_spread(self.profile))
        step = cfg()["simulation"]["step_minutes"]
        start = self.history()["time"].iloc[-1]
        df = pd.DataFrame({"time": start + pd.to_timedelta(np.arange(1, h + 1) * step, unit="min"),
                           "t_out": H["t_out"][0], "outdoor": H["outdoor"][0], "met": H["met"][0]})
        for key in ("tc_lo", "tc_mid", "tc_hi", "psi_lo", "psi_mid", "psi_hi", "hr_mid"):
            df[key] = res[key][0]
        return {"df": df, "p_event": float(res["p_event"][0]), "peak_psi": float(res["peak_psi"][0]),
                "peak_tc": float(res["peak_tc"][0]), "scenario": scenario or Scenario()}

    # ------------------------------------------------------------------ live replay
    def replay(self, t_start: int, hours: int = 24, every_h: int = 1, alert_p: float = 0.5) -> dict:
        """Replay one day as if live: every `every_h` hours the twin absorbs the new wearable readings
        and re-forecasts the rest of the window. Returns frames for animation + when the alert fired."""
        step = cfg()["simulation"]["step_minutes"]
        per_h = 60 // step
        T = self.D["t_out"].shape[1]
        t_end = min(t_start + hours * per_h, T - 1)
        frames_t = list(range(t_start, t_end, every_h * per_h))
        post = F.run_filter(self.profile, self.D, "twin", snapshot_steps=set(frames_t), t_end=t_end + 1,
                            seed=self.seed)
        start = pd.Timestamp(str(self.D["start"][self.i])) if self.D["start"].ndim else pd.Timestamp(str(self.D["start"]))
        times = start + pd.to_timedelta(np.arange(T) * step, unit="min")
        frames = []
        for t in frames_t:
            h = t_end - t
            H, k, _ = build_horizon(self.profile, self.D, t, h)
            r = F.forecast(post["snapshots"][t], k, H, seed=1, spread=F.unlearned_spread(self.profile))
            frames.append({"t": t, "time": times[t], "p_event": float(r["p_event"][0]),
                           "peak_psi": float(r["peak_psi"][0]),
                           "fc": pd.DataFrame({"time": times[t + 1:t_end + 1], "psi_lo": r["psi_lo"][0],
                                               "psi_mid": r["psi_mid"][0], "psi_hi": r["psi_hi"][0],
                                               "tc_mid": r["tc_mid"][0]})})
        alert = next((f for f in frames if f["p_event"] >= alert_p), None)
        sl = slice(t_start, t_end + 1)
        true_psi, true_tc = self.D["true_psi"][0, sl], self.D["true_tc"][0, sl]
        th = cfg()["outcome"]
        hit = (true_psi >= th["psi_threshold"]) | (true_tc >= th["core_threshold_c"])
        danger_time = times[sl][int(np.argmax(hit))] if hit.any() else None
        obs = pd.DataFrame({"time": times[sl], "hr_obs": self.D["hr_obs"][0, sl],
                            "tc_est": post["tc_mean"][0, sl], "tc_sd": post["tc_sd"][0, sl],
                            "true_psi": true_psi, "true_tc": true_tc, "t_out": self.D["t_out"][0, sl],
                            "outdoor": self.D["outdoor"][0, sl]})
        return {"frames": frames, "obs": obs, "alert": alert, "danger_time": danger_time, "alert_p": alert_p}

    def truth(self, hours: int = 24) -> pd.DataFrame:
        """Hidden ground truth for the same window (only for the hide-and-reveal view)."""
        h = hours * 60 // cfg()["simulation"]["step_minutes"]
        sl = slice(self.t + 1, min(self.t + 1 + h, self.D["true_tc"].shape[1]))
        return pd.DataFrame({"tc": self.D["true_tc"][0, sl], "psi": self.D["true_psi"][0, sl]})

    # ------------------------------------------------------------------ explain
    def explain(self, base: dict | None = None) -> list[dict]:
        """Reset one factor at a time; the bigger the drop in risk, the bigger the driver."""
        base = base or self.forecast()
        p = self.profile.iloc[0]
        tests = [("Forecast heat (vs a 35 C day)", "mild_day", None),
                 ("Afternoon outdoor routine", "indoors", Scenario(indoors_from_h=10, indoors_to_h=18)),
                 ("No AC at home", "ac", None) if not p["home_ac"] else None,
                 ("Diuretic-related fluid loss", "diu", Scenario(no_diuretic=True)) if p["diuretic"] else None,
                 ("SGLT2-inhibitor fluid loss", "sg", Scenario(no_sglt2i=True)) if p["sglt2i"] else None]
        out = []
        for item in filter(None, tests):
            label, key, sc = item
            if key == "mild_day":
                peak = self.D["t_out_forecast"][0, self.t + 1:self.t + 289].max()
                sc = Scenario(temp_offset_c=min(0.0, 35.0 - peak))
            elif key == "ac":
                saved = self.profile.copy()
                self.profile.loc[0, "home_ac"] = 1
                r = self.forecast(seed=1)
                self.profile = saved
                out.append({"driver": label, "risk_drop": base["p_event"] - r["p_event"],
                            "peak_psi_drop": base["peak_psi"] - r["peak_psi"]})
                continue
            r = self.forecast(sc, seed=1)
            out.append({"driver": label, "risk_drop": base["p_event"] - r["p_event"],
                        "peak_psi_drop": base["peak_psi"] - r["peak_psi"]})
        st = self.state()
        out.append({"driver": f"Learned heart-rate response {st['hr_gain']:.0f} bpm/C"
                              + (" (beta-blocker blunted)" if p["beta_blocker"] else ""),
                    "risk_drop": np.nan, "peak_psi_drop": np.nan})
        return sorted(out, key=lambda d: -np.nan_to_num(d["peak_psi_drop"], nan=-1))
