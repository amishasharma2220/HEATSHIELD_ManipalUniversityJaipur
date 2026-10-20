"""Physiological Strain Index (Moran, Shitzer & Pandolf, 1998).

PSI = 5 (Tc_t - Tc_0) / (39.5 - Tc_0) + 5 (HR_t - HR_0) / (180 - HR_0), clipped to 0-10.
"""
import numpy as np
from .config import cfg


def psi(tc, hr, tc0, hr0):
    o = cfg()["outcome"]
    tc, hr, tc0, hr0 = (np.asarray(x, float) for x in (tc, hr, tc0, hr0))
    p = 5 * (tc - tc0) / (o["core_max_ref"] - tc0) + 5 * (hr - hr0) / (o["hr_max_ref"] - hr0)
    return np.clip(p, 0, 10)


def psi_temp_half(tc, tc0):
    """Temperature half of PSI (0-5). Still a valid strain signal when heart rate is drug-blunted."""
    o = cfg()["outcome"]
    return np.clip(5 * (np.asarray(tc) - tc0) / (o["core_max_ref"] - tc0), 0, 5)


def is_event(max_psi, max_tc):
    """High heat strain: PSI >= 7, or core temperature >= 39.0 C (so a blunted heart rate can't hide it)."""
    o = cfg()["outcome"]
    return (np.asarray(max_psi) >= o["psi_threshold"]) | (np.asarray(max_tc) >= o["core_threshold_c"])
