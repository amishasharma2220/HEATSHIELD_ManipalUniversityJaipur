"""World simulator: the 'fake reality' that generates wearable data and hidden ground truth.

It is deliberately RICHER than the twin's model (circadian rhythm, humidity-limited sweating,
solar load, per-patient skin offset, fitness, sensor noise and dropouts) and every patient has
hidden traits the twin never sees. The twin must infer them; ground truth is used only to grade it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import cfg, PROCESSED, steps_per_day, dt_hours
from .. import schedule, weather
from ..psi import psi

HIDDEN_COLS = ["sweat_capacity", "hr_gain", "fluid_loss_mult", "fluid_reserve_l", "hr_rest", "tc_rest",
               "skin_offset", "fitness"]


def bsa(weight_kg, height_cm):
    return np.sqrt(np.asarray(weight_kg) * np.asarray(height_cm) / 3600.0)  # Mosteller


def draw_traits(cohort: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    w = cfg()["world"]
    tr = w["traits"]
    med = w["medication"]
    n = len(cohort)
    bb = cohort["beta_blocker"].to_numpy().astype(bool)
    age = cohort["age"].to_numpy()
    age_factor = np.clip(1 - 0.012 * (age - 50), 0.5, 1.0)
    t = pd.DataFrame({"pid": cohort["pid"]})
    t["sweat_capacity"] = np.clip(rng.normal(*tr["sweat_capacity"], n), 0.5, 1.5) * age_factor
    t["hr_gain"] = np.clip(rng.normal(*tr["hr_gain_bpm_per_c"], n), 18, 50) * np.where(bb, med["beta_blocker_hr_gain_mult"], 1.0)
    t["fluid_loss_mult"] = np.clip(rng.normal(*tr["fluid_loss_mult"], n), 0.6, 1.6)
    t["fluid_reserve_l"] = (np.clip(rng.normal(*tr["fluid_reserve_l"], n), 0.8, 2.5)
                            * np.where(cohort["egfr"] < 60, 0.75, 1.0) * cohort["weight_kg"] / 70)
    t["hr_rest"] = np.clip(rng.normal(*tr["hr_rest_bpm"], n), 52, 92) + np.where(bb, med["beta_blocker_rest_hr_shift"], 0)
    t["tc_rest"] = rng.normal(*tr["tc_rest_c"], n)
    t["skin_offset"] = rng.normal(0, 0.3, n)
    t["fitness"] = np.clip(rng.normal(1.0, 0.15, n), 0.6, 1.4)
    return t


def physiology(cohort: pd.DataFrame, traits: pd.DataFrame, X: dict) -> dict:
    """Step the heat + fluid balance for all patients at once. X holds (P, T) input arrays."""
    w = cfg()["world"]
    P, T = X["t_out"].shape
    dt_s = dt_hours() * 3600
    W = cohort["weight_kg"].to_numpy()
    A = bsa(W, cohort["height_cm"].to_numpy())
    age = cohort["age"].to_numpy()
    bb = cohort["beta_blocker"].to_numpy().astype(bool)
    med_extra = (cohort["diuretic"].to_numpy() * w["medication"]["diuretic_extra_l_per_h"]
                 + cohort["sglt2i"].to_numpy() * w["medication"]["sglt2i_extra_l_per_h"]) * traits["fluid_loss_mult"].to_numpy()
    replace = np.clip(w["drinking"]["replace_fraction_young"]
                      - (w["drinking"]["replace_fraction_young"] - w["drinking"]["replace_fraction_old"])
                      * (age - 55) / 30, 0.4, 0.9)
    act_mult = np.where(bb, w["medication"]["beta_blocker_activity_mult"], 1.0) * traits["fitness"].to_numpy()
    tr = {k: traits[k].to_numpy() for k in HIDDEN_COLS}
    heat_cap = W * w["body"]["specific_heat_j_per_kg_k"]
    hours = (np.arange(T) * dt_hours()) % 24

    tc = tr["tc_rest"].copy()
    deficit = np.zeros(P)
    out = {k: np.zeros((P, T)) for k in ("tc", "deficit", "hr", "skin", "sweat_w")}
    for t in range(T):
        circ = w["circadian_amplitude_c"] * np.cos(2 * np.pi * (hours[t] - 17) / 24)
        def_pct = deficit / W * 100
        setpoint = tr["tc_rest"] + circ + w["dehydration"]["core_c_per_pct_bodymass"] * def_pct
        t_eff = np.where(X["outdoor"][:, t], X["t_out"][:, t] + w["solar_load_c"] * X["solar"][:, t] / 900,
                         X["t_in"][:, t])
        skin = np.clip(33 + 0.22 * (t_eff - 28) + 0.45 * (tc - tr["tc_rest"]) + tr["skin_offset"], 30, 37.8)
        m_ex = (X["met"][:, t] - 1) * w["body"]["met_w_per_kg"] * W * 0.8
        h_dry = w["dry_heat_coeff_w_m2_k"] * A * (t_eff - 27)
        e_max = (w["sweat_max_w"] * tr["sweat_capacity"] * (1 - w["humidity_penalty"] * X["rh"][:, t])
                 * (1 - w["dehydration"]["sweat_suppression_at_reserve"] * np.clip(deficit / tr["fluid_reserve_l"], 0, 1.2)))
        e = np.clip(w["sweat_gain_w_per_c"] * (tc - setpoint), 0, np.maximum(e_max, 0))
        r = w["regulatory_loss_w_per_c"] * (tc - setpoint)
        tc = tc + (m_ex + h_dry - e - r) * dt_s / heat_cap
        deficit = deficit + e * dt_s / w["latent_heat_j_per_kg"] + med_extra * dt_hours()
        deficit = np.where(X["drink"][:, t], deficit * (1 - replace), deficit)
        hr = (tr["hr_rest"] + 12 * (X["met"][:, t] - 1) * act_mult + tr["hr_gain"] * (tc - tr["tc_rest"])
              + w["dehydration"]["hr_bpm_per_pct_bodymass"] * def_pct)
        out["tc"][:, t], out["deficit"][:, t], out["hr"][:, t] = tc, deficit, hr
        out["skin"][:, t], out["sweat_w"][:, t] = skin, e
    return out


def build_inputs(cohort: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Weather + actual routines for every patient, as (P, T) arrays."""
    n_days = cfg()["simulation"]["days"]
    T = n_days * steps_per_day()
    P = len(cohort)
    X = {k: np.zeros((P, T)) for k in ("t_out", "rh", "solar", "t_out_forecast")}
    X.update({k: np.zeros((P, T), bool) for k in ("outdoor", "sleep", "drink")})
    X["met"] = np.zeros((P, T))
    starts = []
    for i, row in cohort.iterrows():
        hw = weather.load_city(row["city"])
        first, last = hw["time"].min().normalize(), hw["time"].max().normalize()
        span = (last - first).days - n_days
        start = first + pd.Timedelta(days=int(rng.integers(0, max(span, 1))))
        starts.append(start)
        st = weather.to_steps(hw, start, T)
        X["t_out"][i], X["rh"][i], X["solar"][i] = st["temp_c"], st["rh"], st["solar_wm2"]
        # forecast = truth + a per-day bias (what a real forecast would give the twin)
        bias = np.repeat(rng.normal(0, 1.0, n_days), steps_per_day())
        X["t_out_forecast"][i] = st["temp_c"].to_numpy() + bias
        s = schedule.build(row, n_days, rng, usual=False)
        for k in ("met", "outdoor", "sleep", "drink"):
            X[k][i] = s[k]
    ac = cohort["home_ac"].to_numpy().astype(bool)
    X["t_in"] = weather.indoor_temp(X["t_out"], ac)
    X["t_in_noac"] = weather.indoor_temp(X["t_out"], ac, power_cut=True)
    X["start"] = np.array([str(s.date()) for s in starts])
    return X


def run(cohort: pd.DataFrame | None = None, save: bool = True):
    """Simulate the world. save=False returns (D, traits) in memory (used by the stress tests,
    so the main dataset on disk is never overwritten)."""
    rng = np.random.default_rng(cfg()["seed"] + 1)
    if cohort is None:
        cohort = pd.read_parquet(PROCESSED / "cohort.parquet")
    traits = draw_traits(cohort, rng)
    X = build_inputs(cohort, rng)
    truth = physiology(cohort, traits, X)
    s = cfg()["world"]["sensors"]
    P, T = X["met"].shape
    hr_obs = truth["hr"] + rng.normal(0, s["hr_noise_bpm"], (P, T))
    skin_obs = truth["skin"] + rng.normal(0, s["skin_noise_c"], (P, T))
    drop = rng.random((P, T)) < s["dropout_prob"]
    hr_obs[drop], skin_obs[drop] = np.nan, np.nan
    true_psi = psi(truth["tc"], truth["hr"], traits["tc_rest"].to_numpy()[:, None], traits["hr_rest"].to_numpy()[:, None])
    D = dict(
        # inputs (known to the twin: weather, phone location, logged activity level)
        **{k: X[k] for k in ("t_out", "t_out_forecast", "rh", "solar", "t_in", "t_in_noac",
                             "met", "outdoor", "sleep", "drink", "start")},
        # wearable observations (known to the twin)
        hr_obs=hr_obs, skin_obs=skin_obs, steps=schedule.steps_from_met(X["met"], rng),
        # ground truth (HIDDEN from the twin)
        true_tc=truth["tc"], true_deficit=truth["deficit"], true_hr=truth["hr"], true_psi=true_psi,
    )
    if not save:
        return D, traits
    np.savez_compressed(PROCESSED / "world.npz", **D)
    traits.to_parquet(PROCESSED / "hidden_traits.parquet", index=False)
    print(f"simulated {P} patients x {T} steps")


if __name__ == "__main__":
    run()
