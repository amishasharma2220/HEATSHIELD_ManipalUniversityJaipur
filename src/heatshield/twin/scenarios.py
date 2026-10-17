"""What-if scenarios: build the twin's forecast inputs for the next hours, then modify them.

Medication scenarios are HYPOTHETICAL. They show how much of the modelled risk is attributable
to a drug's fluid effect; they are never a recommendation to stop or change a medication.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from ..config import cfg, steps_per_day
from .. import schedule, weather
from . import model
from .model import Consts


@dataclass
class Scenario:
    name: str = "Baseline"
    temp_offset_c: float = 0.0          # hotter / cooler day than forecast
    power_cut: bool = False             # AC unavailable
    indoors_from_h: float | None = None  # stay indoors and rest between these hours
    indoors_to_h: float | None = None
    activity_scale: float = 1.0         # 0.7 = 30% lighter activity
    extra_fluids: bool = False          # a drink every 90 min while awake
    no_diuretic: bool = False           # HYPOTHETICAL, for clinical judgement only
    no_sglt2i: bool = False             # HYPOTHETICAL, for clinical judgement only

    @property
    def hypothetical(self) -> bool:
        return self.no_diuretic or self.no_sglt2i

    def as_dict(self):
        return asdict(self)


PRESETS = {
    "Baseline": Scenario(),
    "Stay indoors 11am-4pm": Scenario("Stay indoors 11am-4pm", indoors_from_h=11, indoors_to_h=16),
    "Hotter day (+3 C)": Scenario("Hotter day (+3 C)", temp_offset_c=3.0),
    "Power cut (no AC)": Scenario("Power cut (no AC)", power_cut=True),
    "Lighter activity": Scenario("Lighter activity", activity_scale=0.6),
    "Extra fluids": Scenario("Extra fluids", extra_fluids=True),
    "If not on a diuretic (hypothetical)": Scenario("If not on a diuretic (hypothetical)", no_diuretic=True),
}


def build_horizon(cohort: pd.DataFrame, D: dict, t0: int, h: int, scenario: Scenario | None = None,
                  actual: bool = False) -> tuple[dict, Consts, np.ndarray]:
    """Inputs for steps t0+1 .. t0+h for every row of `cohort` (D rows must match cohort rows).

    actual=False: forecast weather + the patient's USUAL routine (what the twin knows in advance).
    actual=True : real weather + logged activity (used for hide-and-reveal evaluation).
    Returns (H inputs, Consts, hour-of-day array).
    """
    sc = scenario or Scenario()
    spd = steps_per_day()
    step_h = cfg()["simulation"]["step_minutes"] / 60
    sl = slice(t0 + 1, t0 + 1 + h)
    hours = (np.arange(t0 + 1, t0 + 1 + h) * step_h) % 24
    ac = cohort["home_ac"].to_numpy().astype(bool)
    if actual:
        t_out = D["t_out"][:, sl] + sc.temp_offset_c
        met, outdoor, drink = D["met"][:, sl].copy(), D["outdoor"][:, sl].copy(), D["drink"][:, sl].copy()
    else:
        t_out = D["t_out_forecast"][:, sl] + sc.temp_offset_c
        n_days = (t0 + 1 + h) // spd + 2
        met, outdoor, drink = [], [], []
        for _, row in cohort.iterrows():
            s = schedule.build(row, n_days, usual=True)
            met.append(s["met"][sl]), outdoor.append(s["outdoor"][sl]), drink.append(s["drink"][sl])
        met, outdoor, drink = np.array(met), np.array(outdoor), np.array(drink)
    # indoor temperature from the full (offset) series so the lag is right, then sliced
    base = (D["t_out"] if actual else D["t_out_forecast"]) + sc.temp_offset_c
    t_in = weather.indoor_temp(base, ac, power_cut=sc.power_cut)[:, sl]
    if sc.indoors_from_h is not None:
        win = (hours >= sc.indoors_from_h) & (hours < sc.indoors_to_h)
        outdoor[:, win] = False
        met[:, win] = np.minimum(met[:, win], 1.3)
    if sc.activity_scale != 1.0:
        met = np.where(met > 1, 1 + (met - 1) * sc.activity_scale, met)
    if sc.extra_fluids:
        awake = (hours >= 8) & (hours < 21)
        extra = awake & (np.round((hours - 8) / 1.5, 6) % 1 == 0)
        drink = drink | extra[None, :]
    H = dict(t_eff=model.effective_temp(t_out, t_in, D["solar"][:, sl], outdoor),
             rh=D["rh"][:, sl], met=met, drink=drink, hour=hours, t_out=t_out, outdoor=outdoor)
    k = Consts.from_cohort(cohort, no_diuretic=sc.no_diuretic, no_sglt2i=sc.no_sglt2i)
    return H, k, hours
