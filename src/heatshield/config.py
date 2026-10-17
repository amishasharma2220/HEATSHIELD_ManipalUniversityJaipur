"""Paths and the single assumptions file."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"
DATA = ROOT / "data"
RAW_EHR = DATA / "raw" / "ehr"
RAW_WEATHER = DATA / "raw" / "weather"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

for p in (RAW_EHR, RAW_WEATHER, PROCESSED, RESULTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

_cfg = None


def cfg() -> dict:
    global _cfg
    if _cfg is None:
        _cfg = yaml.safe_load(CONFIG_PATH.read_text())
    return _cfg


def steps_per_day() -> int:
    return 24 * 60 // cfg()["simulation"]["step_minutes"]


def dt_hours() -> float:
    return cfg()["simulation"]["step_minutes"] / 60.0
