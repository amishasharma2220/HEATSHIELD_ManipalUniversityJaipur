"""Daily routines: activity (MET), indoor/outdoor location, sleep and drinking times.

`usual=True` gives the patient's planned routine (what the twin assumes for a forecast);
`usual=False` adds day-to-day randomness (what actually happens in the world simulator).
"""
from __future__ import annotations

import numpy as np
from .config import cfg, steps_per_day


def _span(hours, a, b):
    return (hours >= a) & (hours < b)


def build(row, n_days: int, rng: np.random.Generator | None = None, usual: bool = False) -> dict:
    spd = steps_per_day()
    step_h = cfg()["simulation"]["step_minutes"] / 60
    hours = np.arange(spd) * step_h
    met, outdoor, sleep, drink = [], [], [], []
    drink_times = cfg()["world"]["drinking"]["times_h"]
    for _ in range(n_days):
        jitter = 0.0 if usual or rng is None else rng.normal(0, 0.25)
        m = np.full(spd, 1.3)
        o = np.zeros(spd, bool)
        s = ~_span(hours, 6.0 + jitter, 22.5 + jitter)
        m[s] = 0.9
        m[_span(hours, 7, 11)] = 1.6
        m[_span(hours, 17, 21)] = 1.4
        if row["morning_walk"] and (usual or rng.random() < 0.85):
            w = _span(hours, 6.25 + jitter, 6.9 + jitter)
            m[w], o[w] = 3.0, True
        if row["outdoor_routine"] and (usual or rng.random() < 0.8):
            start = 11.5 if usual else 11.5 + rng.uniform(-0.75, 1.0)
            dur = 4.0 if usual else rng.uniform(2.5, 4.5)
            w = _span(hours, start, start + dur)
            m[w], o[w] = 2.2, True
        if row["morning_walk"] and not row["outdoor_routine"] and (usual or rng.random() < 0.5):
            w = _span(hours, 18.0, 18.6)
            m[w], o[w] = 2.5, True
        d = np.zeros(spd, bool)
        for t in drink_times:
            d[int(round(t / step_h))] = True
        met.append(m), outdoor.append(o), sleep.append(s), drink.append(d)
    return {k: np.concatenate(v) for k, v in
            dict(met=met, outdoor=outdoor, sleep=sleep, drink=drink).items()}


def steps_from_met(met: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = np.clip((met - 1.2) * 140, 0, None)
    return np.round(np.clip(base * rng.lognormal(0, 0.25, size=met.shape), 0, 900)).astype(int)
