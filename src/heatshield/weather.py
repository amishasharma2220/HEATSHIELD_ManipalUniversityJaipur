"""Weather: real Open-Meteo history when available, a labelled synthetic fallback otherwise.

Run `python -m heatshield.weather --fetch` on a machine with internet to download the
May-June 2024 North-India heatwave for five cities. The pipeline then uses the real data.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .config import cfg, RAW_WEATHER

CITIES = {
    "Jaipur": (26.91, 75.79),
    "Delhi": (28.61, 77.21),
    "Jodhpur": (26.24, 73.02),
    "Nagpur": (21.15, 79.09),
    "Lucknow": (26.85, 80.95),
}
PERIOD = ("2024-05-15", "2024-06-04")   # documented 2024 North-India heatwave window
SOURCE_FILE = RAW_WEATHER / "SOURCE.txt"


def fetch_open_meteo() -> None:
    """Download hourly temperature, humidity and solar radiation from the Open-Meteo archive."""
    for city, (lat, lon) in CITIES.items():
        q = urlencode(dict(latitude=lat, longitude=lon, start_date=PERIOD[0], end_date=PERIOD[1],
                           hourly="temperature_2m,relative_humidity_2m,shortwave_radiation",
                           timezone="Asia/Kolkata"))
        with urllib.request.urlopen(f"https://archive-api.open-meteo.com/v1/archive?{q}", timeout=60) as r:
            h = json.load(r)["hourly"]
        pd.DataFrame({"time": pd.to_datetime(h["time"]), "temp_c": h["temperature_2m"],
                      "rh": np.array(h["relative_humidity_2m"], float) / 100.0,
                      "solar_wm2": h["shortwave_radiation"]}).to_csv(RAW_WEATHER / f"{city}.csv", index=False)
        print("saved", city)
    SOURCE_FILE.write_text(f"REAL: Open-Meteo historical archive, {PERIOD[0]} to {PERIOD[1]}\n")


def generate_synthetic(seed: int = 7) -> None:
    """Fallback: a plausible pre-monsoon heatwave shape. Clearly labelled as synthetic."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(PERIOD[0], PERIOD[1], freq="D")
    nd = len(days)
    # heatwave builds to a peak in the middle of the window then eases
    profile = 40.5 + 5.5 * np.exp(-((np.arange(nd) - nd * 0.55) / (nd * 0.25)) ** 2)
    offsets = {"Jaipur": 0.0, "Delhi": 0.3, "Jodhpur": 1.2, "Nagpur": 0.8, "Lucknow": -0.6}
    hours = np.arange(24)
    shape = np.where(hours < 15, (1 - np.cos(np.pi * np.clip((hours - 5.5) / 9.5, 0, 1))) / 2,
                     (1 + np.cos(np.pi * np.clip((hours - 15) / 14.5, 0, 1))) / 2)
    for city, off in offsets.items():
        rows = []
        for d, day in enumerate(days):
            tmax = profile[d] + off + rng.normal(0, 1.0)
            tmin = tmax - rng.uniform(12, 15)
            rh_day = np.clip(rng.normal(0.22, 0.06), 0.08, 0.5)
            cloud = rng.uniform(0.85, 1.0)
            for h in hours:
                t = tmin + (tmax - tmin) * shape[h]
                rh = np.clip(rh_day * (1 + 0.9 * (1 - shape[h])), 0.05, 0.9)
                sol = max(0.0, 950 * np.sin(np.pi * (h - 6) / 13)) * cloud if 6 <= h <= 19 else 0.0
                rows.append((day + pd.Timedelta(hours=int(h)), round(t, 1), round(rh, 3), round(sol, 0)))
        pd.DataFrame(rows, columns=["time", "temp_c", "rh", "solar_wm2"]).to_csv(RAW_WEATHER / f"{city}.csv",
                                                                                  index=False)
    SOURCE_FILE.write_text("SYNTHETIC placeholder (no internet when built). "
                           "Run `python -m heatshield.weather --fetch` to replace with real Open-Meteo data.\n")


def weather_source() -> str:
    return SOURCE_FILE.read_text().strip() if SOURCE_FILE.exists() else "missing"


def load_city(city: str) -> pd.DataFrame:
    path = RAW_WEATHER / f"{city}.csv"
    if not path.exists():
        generate_synthetic()
    return pd.read_csv(path, parse_dates=["time"])


def to_steps(hourly: pd.DataFrame, start: pd.Timestamp, n_steps: int) -> pd.DataFrame:
    """Interpolate hourly weather to simulation steps starting at `start` (midnight)."""
    step = cfg()["simulation"]["step_minutes"]
    idx = pd.date_range(start, periods=n_steps, freq=f"{step}min")
    h = hourly.set_index("time").sort_index()
    h = h[~h.index.duplicated()]
    out = h.reindex(h.index.union(idx)).interpolate("time").reindex(idx)
    return out.bfill().ffill()


def indoor_temp(outdoor: np.ndarray, home_ac: bool | np.ndarray, power_cut: bool = False) -> np.ndarray:
    """Indoor temperature: lagged, damped outdoor curve; capped when AC is available.

    `outdoor` is a 1-D series in simulation steps (or 2-D, patients x steps).
    """
    c = cfg()["indoor"]
    lag = int(c["lag_hours"] * 60 / cfg()["simulation"]["step_minutes"])
    out = np.atleast_2d(outdoor)
    lagged = np.concatenate([np.repeat(out[:, :1], lag, axis=1), out[:, :-lag]], axis=1)
    spd = 24 * 60 // cfg()["simulation"]["step_minutes"]
    kernel = np.ones(spd) / spd
    daily_mean = np.vstack([np.convolve(np.pad(r, (spd // 2, spd - spd // 2 - 1), mode="edge"), kernel, "valid")
                            for r in lagged])
    t_in = daily_mean + c["damping"] * (lagged - daily_mean) + c["offset_c"]
    if not power_cut:
        ac = np.broadcast_to(np.atleast_1d(home_ac).astype(bool)[:, None], t_in.shape)
        t_in = np.where(ac, np.minimum(t_in, c["ac_cap_c"]), t_in)
    return t_in if np.ndim(outdoor) == 2 else t_in[0]


def heat_index_c(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """NOAA Rothfusz heat index, returned in deg C (inputs deg C, RH as fraction)."""
    t = np.asarray(t_c) * 9 / 5 + 32
    r = np.asarray(rh) * 100
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * r - 0.22475541 * t * r - 6.83783e-3 * t * t
          - 5.481717e-2 * r * r + 1.22874e-3 * t * t * r + 8.5282e-4 * t * r * r - 1.99e-6 * t * t * r * r)
    hi = np.where(t < 80, 0.5 * (t + 61 + (t - 68) * 1.2 + r * 0.094), hi)
    return (hi - 32) * 5 / 9


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download real data from Open-Meteo")
    a = ap.parse_args()
    fetch_open_meteo() if a.fetch else generate_synthetic()
    print(weather_source())
