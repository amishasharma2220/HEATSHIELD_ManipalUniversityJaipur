"""State estimation: the HEATSHIELD particle filter, plus the Buller (2013) baseline.

The particle filter tracks hidden state (core temperature, fluid deficit) AND learns personal
traits (sweat capacity, HR gain, fluid-loss multiplier, resting HR) at the same time. It is
vectorised over all patients and particles: arrays are (P, N).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import cfg
from ..psi import psi, is_event
from . import model
from .model import Consts, PARAMS, TC_REST

# ----------------------------------------------------------------------------- Buller baseline
# Buller et al. (2013) Physiol. Meas. 34:781 — core temperature from heart rate alone.
# Coefficients b2=-4.5714, b1=384.4286, b0=-7887.1, gamma^2=0.022^2 per min, sigma=18.88 as in
# Buller et al. (2013) and US patent US20140180027A1; initial CT 37.1 C as in later re-implementations.
BULLER = dict(a=-4.5714, b=384.4286, c=-7887.1, gamma_per_min=0.022 ** 2, sigma=18.88, x0=37.1)


def buller_filter(hr_obs: np.ndarray) -> np.ndarray:
    """Extended Kalman filter, random-walk core temperature, quadratic HR observation. (P, T) in, (P, T) out."""
    P, T = hr_obs.shape
    q = BULLER["gamma_per_min"] * cfg()["simulation"]["step_minutes"]
    x, v = np.full(P, BULLER["x0"]), np.zeros(P)
    out = np.zeros((P, T))
    for t in range(T):
        v = v + q
        y = hr_obs[:, t]
        ok = ~np.isnan(y)
        cg = 2 * BULLER["a"] * x + BULLER["b"]
        k = v * cg / (cg ** 2 * v + BULLER["sigma"] ** 2)
        pred = BULLER["a"] * x ** 2 + BULLER["b"] * x + BULLER["c"]
        x = np.where(ok, x + k * (np.nan_to_num(y) - pred), x)
        v = np.where(ok, (1 - k * cg) * v, v)
        out[:, t] = x
    return out


# ----------------------------------------------------------------------------- particle filter
def _systematic_resample(w: np.ndarray, rng) -> np.ndarray:
    """Row-wise systematic resampling. w is (P, N), rows sum to 1. Returns indices (P, N)."""
    P, N = w.shape
    cum = np.cumsum(w, axis=1)
    cum[:, -1] = 1.0
    pos = (rng.random((P, 1)) + np.arange(N)[None, :]) / N
    offs = np.arange(P)[:, None]
    idx = np.searchsorted((cum + offs).ravel(), (pos + offs).ravel(), side="left").reshape(P, N) - offs * N
    return np.clip(idx, 0, N - 1)


def inputs_from_world(D: dict, rows=slice(None), use_forecast=False, power_cut=False) -> dict:
    """Twin-visible inputs: weather (+ indoor), phone location, activity from steps, drink times."""
    t_out = D["t_out_forecast"][rows] if use_forecast else D["t_out"][rows]
    t_in = D["t_in_noac"][rows] if power_cut else D["t_in"][rows]
    return dict(t_eff=model.effective_temp(t_out, t_in, D["solar"][rows], D["outdoor"][rows]),
                rh=D["rh"][rows], met=D["met"][rows], drink=D["drink"][rows])


def init_cloud(cohort: pd.DataFrame, hr_obs: np.ndarray, mode: str, rng, N: int) -> dict:
    P = len(cohort)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            night = np.nanmedian(hr_obs[:, :60], axis=1)              # first 5 h of data = night 1
    # watch not worn on night 1 -> fall back to the EHR-only resting-HR prior
    night = np.where(np.isnan(night), np.where(cohort["beta_blocker"].to_numpy() == 1, 58, 66), night)
    if mode == "population":
        point = model.population_point(cohort)
        th = {k: np.repeat(point[k][:, None], N, 1) for k in PARAMS}
    else:
        pri = model.ehr_priors(cohort, night)
        if mode == "ehr":
            th = {k: np.repeat(pri[k][0][:, None], N, 1) for k in PARAMS}
        else:
            learned = cfg()["twin"].get("learned_params", PARAMS)
            th = {k: pri[k][0][:, None] + (pri[k][1][:, None] if k in learned else 0) * rng.standard_normal((P, N))
                  for k in PARAMS}
            th["sweat_capacity"] = np.clip(th["sweat_capacity"], 0.25, 1.6)
            th["hr_gain"] = np.clip(th["hr_gain"], 4, 60)
            th["fluid_loss_mult"] = np.clip(th["fluid_loss_mult"], 0.3, 2.0)
    return dict(tc=TC_REST + 0.15 * rng.standard_normal((P, N)), deficit=np.zeros((P, N)), **th)


def run_filter(cohort: pd.DataFrame, D: dict, mode: str = "twin", snapshot_steps=(), t_end=None,
               seed: int = 0, N: int | None = None) -> dict:
    """Filter every patient through the wearable stream.

    mode: "twin" (EHR priors + learning), "ehr" (EHR priors, no learning),
          "population" (one-size-fits-all, no medication awareness, no learning).
    Returns posterior summaries per step and equal-weight particle clouds at `snapshot_steps`.
    """
    tw = cfg()["twin"]
    N = N or tw["n_particles"]
    rng = np.random.default_rng(seed)
    T = D["hr_obs"].shape[1] if t_end is None else t_end
    X = inputs_from_world(D)
    k = Consts.from_cohort(cohort)
    if mode == "population":   # ignores the medication list entirely
        k.act_mult[:] = 1.0
        k.med_extra[:] = 0.04
    kk = Consts(**{f: getattr(k, f)[:, None] for f in k.__dataclass_fields__})
    cloud = init_cloud(cohort, D["hr_obs"], mode, rng, N)
    P = len(cohort)
    logw = np.zeros((P, N))
    learn = mode == "twin"
    a = tw.get("liu_west_a", 0.995)                       # Liu-West shrinkage
    learned = [p for p in PARAMS if p in tw.get("learned_params", PARAMS)]
    out = {k_: np.zeros((P, T)) for k_ in ("tc_mean", "tc_sd", "def_mean")}
    out["params"] = {p: np.zeros((P, T)) for p in PARAMS}
    out["params_sd"] = {p: np.zeros((P, T)) for p in PARAMS}
    snaps = {}
    snapshot_steps = set(snapshot_steps)
    hours = (np.arange(T) * cfg()["simulation"]["step_minutes"] / 60) % 24
    for t in range(T):
        th = {p: cloud[p] for p in PARAMS}
        noise = (tw["process_noise_tc"] * rng.standard_normal((P, N)),
                 tw["process_noise_deficit"] * rng.standard_normal((P, N)))
        cloud["tc"], cloud["deficit"] = model.step(cloud["tc"], cloud["deficit"], th, kk, X["met"][:, t:t + 1],
                                                   X["t_eff"][:, t:t + 1], X["rh"][:, t:t + 1],
                                                   X["drink"][:, t:t + 1], hours[t], noise)
        hr_p, sk_p = model.observe(cloud["tc"], cloud["deficit"], th, kk, X["met"][:, t:t + 1], X["t_eff"][:, t:t + 1])
        y_hr, y_sk = D["hr_obs"][:, t:t + 1], D["skin_obs"][:, t:t + 1]
        ll = np.where(np.isnan(y_hr), 0, -0.5 * ((np.nan_to_num(y_hr) - hr_p) / tw["obs_noise_hr"]) ** 2)
        ll += np.where(np.isnan(y_sk), 0, -0.5 * ((np.nan_to_num(y_sk) - sk_p) / tw["obs_noise_skin"]) ** 2)
        logw = logw + ll
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        mean_tc = (w * cloud["tc"]).sum(1)
        out["tc_mean"][:, t] = mean_tc
        out["tc_sd"][:, t] = np.sqrt((w * (cloud["tc"] - mean_tc[:, None]) ** 2).sum(1))
        out["def_mean"][:, t] = (w * cloud["deficit"]).sum(1)
        for p in PARAMS:
            m = (w * cloud[p]).sum(1)
            out["params"][p][:, t] = m
            out["params_sd"][p][:, t] = np.sqrt((w * (cloud[p] - m[:, None]) ** 2).sum(1))
        ess = 1.0 / (w ** 2).sum(1)
        need = (ess < N * tw.get("resample_ess_frac", 0.5)) | (t in snapshot_steps)
        if need.any():
            idx = _systematic_resample(w, rng)
            rows = np.where(need)[0]
            for key in cloud:
                cloud[key][rows] = np.take_along_axis(cloud[key][rows], idx[rows], axis=1)
            logw[rows] = 0
            if learn:
                for p in learned:
                    v = cloud[p][rows]
                    mu, sd = v.mean(1, keepdims=True), v.std(1, keepdims=True)
                    h = np.sqrt(1 - a ** 2)
                    cloud[p][rows] = a * v + (1 - a) * mu + h * sd * rng.standard_normal(v.shape)
                cloud["sweat_capacity"] = np.clip(cloud["sweat_capacity"], 0.25, 1.6)
                cloud["hr_gain"] = np.clip(cloud["hr_gain"], 4, 60)
                cloud["fluid_loss_mult"] = np.clip(cloud["fluid_loss_mult"], 0.3, 2.0)
        if t in snapshot_steps:
            snaps[t] = {key: v.copy() for key, v in cloud.items()}
    out["snapshots"] = snaps
    out["consts"] = k
    return out


# ----------------------------------------------------------------------------- forecasting
def unlearned_spread(cohort: pd.DataFrame) -> dict:
    """EHR-prior sd for traits the twin cannot learn from wrist data; propagated into forecasts."""
    learned = cfg()["twin"].get("learned_params", PARAMS)
    pri = model.ehr_priors(cohort)
    return {p: pri[p][1] for p in PARAMS if p not in learned}


CLIPS = {"sweat_capacity": (0.25, 1.6), "hr_gain": (4, 60), "fluid_loss_mult": (0.3, 2.0), "hr_rest": (40, 110)}


def forecast(cloud: dict, k: Consts, H: dict, seed: int = 1, spread: dict | None = None) -> dict:
    """Run every particle forward through horizon inputs H (each (P, h)). Equal-weight cloud in.

    `spread` (from unlearned_spread) adds prior uncertainty for traits the filter cannot learn,
    so the forecast band reflects what the twin genuinely does not know.
    """
    tw = cfg()["twin"]
    rng = np.random.default_rng(seed)
    kk = Consts(**{f: getattr(k, f)[:, None] for f in k.__dataclass_fields__})
    tc, deficit = cloud["tc"].copy(), cloud["deficit"].copy()
    th = {p: cloud[p] for p in PARAMS}
    P, N = tc.shape
    for p, sd in (spread or {}).items():
        th[p] = np.clip(th[p] + np.asarray(sd)[:, None] * rng.standard_normal((P, N)), *CLIPS[p])
    h = H["met"].shape[1]
    max_psi, max_tc = np.zeros((P, N)), np.full((P, N), -np.inf)
    q = (5, 50, 95)
    res = {f"{v}_{s}": np.zeros((P, h)) for v in ("tc", "psi", "hr") for s in ("lo", "mid", "hi")}
    res["tc_mean"], res["psi_mean"] = np.zeros((P, h)), np.zeros((P, h))
    sd_tc = tw.get("forecast_noise_tc", tw["process_noise_tc"])
    for t in range(h):
        noise = (sd_tc * rng.standard_normal((P, N)),
                 tw["process_noise_deficit"] * rng.standard_normal((P, N)))
        tc, deficit = model.step(tc, deficit, th, kk, H["met"][:, t:t + 1], H["t_eff"][:, t:t + 1],
                                 H["rh"][:, t:t + 1], H["drink"][:, t:t + 1], H["hour"][t], noise)
        hr, _ = model.observe(tc, deficit, th, kk, H["met"][:, t:t + 1], H["t_eff"][:, t:t + 1])
        ps = psi(tc, hr, TC_REST, th["hr_rest"])
        max_psi, max_tc = np.maximum(max_psi, ps), np.maximum(max_tc, tc)
        for v, arr in (("tc", tc), ("psi", ps), ("hr", hr)):
            lo, mid, hi = np.percentile(arr, q, axis=1)
            res[f"{v}_lo"][:, t], res[f"{v}_mid"][:, t], res[f"{v}_hi"][:, t] = lo, mid, hi
        res["tc_mean"][:, t], res["psi_mean"][:, t] = tc.mean(1), ps.mean(1)
    res["p_event"] = is_event(max_psi, max_tc).mean(1)
    res["peak_psi"] = max_psi.mean(1)
    res["peak_tc"] = max_tc.mean(1)
    return res
