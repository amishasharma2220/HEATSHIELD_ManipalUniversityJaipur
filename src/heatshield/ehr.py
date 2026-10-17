"""Synthetic EHR.

Two paths produce the SAME cohort table:
  1. `generate_synthea_style()` writes Synthea-format CSVs (patients, conditions,
     medications, observations) with India-relevant defaults.
  2. Real Synthea output can be dropped into data/raw/ehr/ instead.
`load_cohort()` reads either and maps it to one row per patient.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd

from .config import cfg, RAW_EHR, PROCESSED

# Keyword maps from free-text Synthea descriptions to our flags
CONDITION_KEYS = {
    "hypertension": ["hypertension"],
    "t2d": ["diabetes mellitus type 2", "type 2 diabetes", "prediabetes"],
    "ckd": ["chronic kidney disease"],
}
MED_KEYS = {
    "beta_blocker": ["metoprolol", "atenolol", "bisoprolol", "carvedilol", "propranolol", "nebivolol"],
    "diuretic": ["hydrochlorothiazide", "chlorthalidone", "furosemide", "torsemide", "indapamide"],
    "sglt2i": ["dapagliflozin", "empagliflozin", "canagliflozin"],
    "acei_arb": ["lisinopril", "ramipril", "enalapril", "losartan", "telmisartan", "valsartan", "olmesartan"],
}
MED_NAMES = {
    "beta_blocker": ["Metoprolol succinate 50 MG", "Atenolol 50 MG"],
    "diuretic": ["Hydrochlorothiazide 25 MG", "Chlorthalidone 12.5 MG"],
    "sglt2i": ["Dapagliflozin 10 MG", "Empagliflozin 10 MG"],
    "acei_arb": ["Telmisartan 40 MG", "Ramipril 5 MG"],
    "other": ["Metformin 500 MG", "Amlodipine 5 MG", "Atorvastatin 20 MG"],
}
FIRST = ["Ramesh", "Sunita", "Mohan", "Kamla", "Suresh", "Geeta", "Vijay", "Asha", "Prakash", "Meena",
         "Rajendra", "Usha", "Anil", "Savitri", "Gopal", "Lata", "Mahesh", "Pushpa", "Dinesh", "Saroj"]
LAST = ["Sharma", "Verma", "Gupta", "Meena", "Jain", "Singh", "Agarwal", "Choudhary", "Yadav", "Joshi"]
CITIES = ["Jaipur", "Delhi", "Jodhpur", "Nagpur", "Lucknow"]


def generate_synthea_style(seed: int | None = None) -> None:
    """Write Synthea-format CSVs for a cohort of older adults with chronic conditions."""
    c = cfg()["cohort"]
    rng = np.random.default_rng(seed if seed is not None else cfg()["seed"])
    n = c["n_patients"]
    patients, conditions, meds, obs = [], [], [], []
    for i in range(n):
        pid = str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | i))
        sex = rng.choice(["M", "F"])
        age = int(rng.integers(c["age_min"], c["age_max"] + 1))
        weight = float(np.clip(rng.normal(72 if sex == "M" else 62, 10), 42, 110))
        height = float(np.clip(rng.normal(166 if sex == "M" else 153, 7), 140, 190))
        patients.append(dict(Id=pid, BIRTHDATE=f"{2026 - age}-06-15", GENDER=sex,
                             FIRST=rng.choice(FIRST), LAST=rng.choice(LAST),
                             CITY=rng.choice(CITIES), STATE="IN",
                             HOME_AC=int(rng.random() < c["p_home_ac"]),
                             OUTDOOR_ROUTINE=int(rng.random() < c["p_outdoor_routine"]),
                             MORNING_WALK=int(rng.random() < 0.5)))
        htn = rng.random() < c["p_hypertension"]
        t2d = rng.random() < c["p_t2d"]
        ckd = rng.random() < c["p_ckd"] * (1.4 if age > 70 else 1.0)
        for flag, name in ((htn, "Essential hypertension (disorder)"),
                           (t2d, "Diabetes mellitus type 2 (disorder)"),
                           (ckd, "Chronic kidney disease stage 3 (disorder)")):
            if flag:
                conditions.append(dict(PATIENT=pid, START="2018-01-01", DESCRIPTION=name))
        bb = rng.random() < c["p_beta_blocker"]
        diu = htn and rng.random() < c["p_diuretic"] / max(c["p_hypertension"], 1e-6)
        sg = t2d and rng.random() < c["p_sglt2i"] / max(c["p_t2d"], 1e-6)
        ace = htn and rng.random() < c["p_acei_arb"] / max(c["p_hypertension"], 1e-6)
        for flag, cls in ((bb, "beta_blocker"), (diu, "diuretic"), (sg, "sglt2i"), (ace, "acei_arb")):
            if flag:
                meds.append(dict(PATIENT=pid, START="2023-01-01", STOP="", DESCRIPTION=rng.choice(MED_NAMES[cls])))
        if t2d:
            meds.append(dict(PATIENT=pid, START="2022-01-01", STOP="", DESCRIPTION="Metformin 500 MG"))
        egfr = float(np.clip(rng.normal(48 if ckd else 82, 10) - 0.4 * (age - 60), 18, 110))
        sbp = float(np.clip(rng.normal(142 if htn else 126, 12), 100, 190))
        for code, desc, val, unit in (("33914-3", "Glomerular filtration rate/1.73 sq M.predicted", egfr, "mL/min"),
                                      ("29463-7", "Body Weight", weight, "kg"),
                                      ("8302-2", "Body Height", height, "cm"),
                                      ("8480-6", "Systolic Blood Pressure", sbp, "mm[Hg]")):
            obs.append(dict(PATIENT=pid, DATE="2026-03-01", CODE=code, DESCRIPTION=desc,
                            VALUE=round(val, 1), UNITS=unit))
    pd.DataFrame(patients).to_csv(RAW_EHR / "patients.csv", index=False)
    pd.DataFrame(conditions).to_csv(RAW_EHR / "conditions.csv", index=False)
    pd.DataFrame(meds).to_csv(RAW_EHR / "medications.csv", index=False)
    pd.DataFrame(obs).to_csv(RAW_EHR / "observations.csv", index=False)


def _has(text: str, keys: list[str]) -> bool:
    t = str(text).lower()
    return any(k in t for k in keys)


def load_cohort() -> pd.DataFrame:
    """Map Synthea-format CSVs to one row per patient with the flags the twin needs."""
    p = pd.read_csv(RAW_EHR / "patients.csv")
    cond = pd.read_csv(RAW_EHR / "conditions.csv")
    med = pd.read_csv(RAW_EHR / "medications.csv")
    obs = pd.read_csv(RAW_EHR / "observations.csv")
    if "STOP" in med:
        med = med[med["STOP"].isna() | (med["STOP"].astype(str) == "")]

    df = pd.DataFrame({"patient_id": p["Id"]})
    df["name"] = p["FIRST"].str.replace(r"\d", "", regex=True) + " " + p["LAST"].str.replace(r"\d", "", regex=True)
    df["sex"] = p["GENDER"]
    df["age"] = 2026 - pd.to_datetime(p["BIRTHDATE"]).dt.year
    df["city"] = p.get("CITY", "Jaipur")
    for col, default in (("HOME_AC", 0), ("OUTDOOR_ROUTINE", 0), ("MORNING_WALK", 0)):
        df[col.lower()] = p[col].astype(int) if col in p else default
    for flag, keys in CONDITION_KEYS.items():
        ids = set(cond.loc[cond["DESCRIPTION"].map(lambda d: _has(d, keys)), "PATIENT"])
        df[flag] = df["patient_id"].isin(ids).astype(int)
    for flag, keys in MED_KEYS.items():
        ids = set(med.loc[med["DESCRIPTION"].map(lambda d: _has(d, keys)), "PATIENT"])
        df[flag] = df["patient_id"].isin(ids).astype(int)
    meds_text = med.groupby("PATIENT")["DESCRIPTION"].apply(lambda s: "; ".join(sorted(set(s))))
    df["medications"] = df["patient_id"].map(meds_text).fillna("")

    def latest(code):
        o = obs[obs["CODE"].astype(str) == code].sort_values("DATE")
        return o.groupby("PATIENT")["VALUE"].last().astype(float)

    df["egfr"] = df["patient_id"].map(latest("33914-3")).fillna(75.0)
    df["weight_kg"] = df["patient_id"].map(latest("29463-7")).fillna(68.0)
    df["height_cm"] = df["patient_id"].map(latest("8302-2")).fillna(160.0)
    df["sbp"] = df["patient_id"].map(latest("8480-6")).fillna(135.0)
    df = df[df["age"] >= cfg()["cohort"]["age_min"]].reset_index(drop=True)
    df.insert(0, "pid", [f"P{i:03d}" for i in range(len(df))])
    df.to_parquet(PROCESSED / "cohort.parquet", index=False)
    return df


if __name__ == "__main__":
    if not (RAW_EHR / "patients.csv").exists():
        generate_synthea_style()
    print(load_cohort().head())
