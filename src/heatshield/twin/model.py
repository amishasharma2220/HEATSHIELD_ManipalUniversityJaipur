"""The twin's own physiological model: a SIMPLER, approximate heat + fluid balance.

Deliberate differences from the world simulator (so the twin cannot read its own answers back):
a fixed textbook circadian term instead of a personal one, different sweat/regulation gains, smaller solar term, fixed 37.0 C resting core,
coarser skin-temperature model. Personal traits are not known; they are particle parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..config import cfg, dt_hours

TC_REST = 37.0
PARAMS = ("sweat_capacity", "hr_gain", "fluid_loss_mult", "hr_rest")


@dataclass
class Consts:
    """Per-patient constants the twin reads from the EHR (arrays of shape (P,))."""
    weight: np.ndarray
    bsa: np.ndarray
    reserve: np.ndarray
    replace: np.ndarray
    act_mult: np.ndarray
    med_extra: np.ndarray          # L/h from diuretic + SGLT2i before personal scaling
    beta_blocker: np.ndarray

    @classmethod
    def from_cohort(cls, c: pd.DataFrame, no_diuretic=False, no_sglt2i=False) -> "Consts":
        W = c["weight_kg"].to_numpy(float)
        bb = c["beta_blocker"].to_numpy().astype(bool)
        age = c["age"].to_numpy(float)
        diu = 0 if no_diuretic else c["diuretic"].to_numpy()
        sg = 0 if no_sglt2i else c["sglt2i"].to_numpy()
        return cls(weight=W, bsa=np.sqrt(W * c["height_cm"].to_numpy(float) / 3600),
                   reserve=1.6 * np.where(c["egfr"] < 60, 0.75, 1.0) * W / 70,
                   replace=np.clip(0.8 - 0.25 * (age - 55) / 30, 0.45, 0.85),
                   act_mult=np.where(bb, 0.7, 1.0), med_extra=0.03 * diu + 0.02 * sg, beta_blocker=bb)

    def take(self, idx) -> "Consts":
        return Consts(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})


def ehr_priors(c: pd.DataFrame, night_hr: np.ndarray | None = None) -> dict:
    """Prior mean and sd for each personal trait, set from the EHR (step 1 of the twin)."""
    pr = cfg()["twin"]["priors"]
    age = c["age"].to_numpy(float)
    bb = c["beta_blocker"].to_numpy().astype(bool)
    has_med = (c["diuretic"].to_numpy() + c["sglt2i"].to_numpy()) > 0
    hr_rest_mu = night_hr + 4 if night_hr is not None else np.where(bb, 62, 70)
    return {
        "sweat_capacity": (pr["sweat_capacity"][0] * np.clip(1 - 0.01 * (age - 50), 0.55, 1.0), np.full(len(c), pr["sweat_capacity"][1])),
        "hr_gain": (np.where(bb, 0.5, 1.0) * pr["hr_gain_bpm_per_c"][0], np.where(bb, 0.6, 1.0) * pr["hr_gain_bpm_per_c"][1]),
        "fluid_loss_mult": (np.full(len(c), pr["fluid_loss_mult"][0]), np.where(has_med, 0.3, 0.2)),
        "hr_rest": (hr_rest_mu, np.full(len(c), 5.0)),
    }


def population_point(c: pd.DataFrame) -> dict:
    """One-size-fits-all model: average elderly traits, no medication awareness."""
    n = len(c)
    return {"sweat_capacity": np.full(n, 0.75), "hr_gain": np.full(n, 32.0),
            "fluid_loss_mult": np.full(n, 1.0), "hr_rest": np.full(n, 70.0)}


def effective_temp(t_out, t_in, solar, outdoor):
    return np.where(outdoor, t_out + 3.0 * solar / 900, t_in)


def circadian(hour):
    """Textbook circadian core-temperature rhythm (fixed, not personalised): +/-0.2 C, peak 16:00."""
    return 0.2 * np.cos(2 * np.pi * (np.asarray(hour) - 16) / 24)


def step(tc, deficit, th: dict, k: Consts, met, t_eff, rh, drink, hour, noise=None):
    """One time step for arrays shaped (P, N) (patients x particles). k fields are (P, 1)."""
    dt_s = dt_hours() * 3600
    def_pct = deficit / k.weight * 100
    setpoint = TC_REST + circadian(hour) + 0.1 * def_pct
    m_ex = (met - 1) * 1.16 * k.weight * 0.8
    h_dry = 8.0 * k.bsa * (t_eff - 27)
    e_max = 400 * th["sweat_capacity"] * (1 - 0.5 * rh) * (1 - 0.5 * np.clip(deficit / k.reserve, 0, 1.2))
    e = np.clip(430 * (tc - setpoint), 0, np.maximum(e_max, 0))
    r = 110 * (tc - setpoint)
    tc = tc + (m_ex + h_dry - e - r) * dt_s / (k.weight * 3490)
    deficit = deficit + e * dt_s / 2.43e6 + k.med_extra * th["fluid_loss_mult"] * dt_hours()
    deficit = np.where(drink, deficit * (1 - k.replace), deficit)
    if noise is not None:
        tc = tc + noise[0]
        deficit = np.maximum(deficit + noise[1], 0)
    return tc, deficit


def observe(tc, deficit, th: dict, k: Consts, met, t_eff):
    """What the wearable should read if the particle were true."""
    def_pct = deficit / k.weight * 100
    hr = th["hr_rest"] + 12 * (met - 1) * k.act_mult + th["hr_gain"] * (tc - TC_REST) + 3 * def_pct
    skin = np.clip(33 + 0.2 * (t_eff - 28) + 0.5 * (tc - TC_REST), 30, 37.8)
    return hr, skin
