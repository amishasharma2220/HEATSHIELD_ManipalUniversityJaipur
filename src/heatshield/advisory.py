"""Draft heat-safety messages for a patient's family, in Hindi and English.

The text is a DRAFT for the doctor to review, edit and approve. It never advises changing a medicine,
and it defers fluid advice to the doctor because some patients (e.g. heart failure on diuretics) are on a
fluid limit.
"""
from __future__ import annotations

import pandas as pd


def _fmt(t) -> str:
    return pd.Timestamp(t).strftime("%I:%M %p").lstrip("0")


def draft(row: pd.Series, p_event: float, window: tuple | None, day) -> dict:
    """row: cohort row. window: (start, end) of the high-risk period or None. Returns {'en','hi','level'}."""
    level = "high" if p_event >= 0.5 else "moderate" if p_event >= 0.2 else "low"
    first = str(row["name"]).split(" ")[0]
    date = pd.Timestamp(day).strftime("%d %b")
    if level == "low":
        return {"level": level,
                "en": f"Namaste. {first} ji's heat risk for {date} is low. Usual precautions: stay in shade in the "
                      f"afternoon and keep drinking water as advised by the doctor.",
                "hi": f"नमस्ते। {date} को {first} जी के लिए गर्मी का खतरा कम है। सामान्य सावधानी रखें: दोपहर में छाँव "
                      f"में रहें और डॉक्टर की बताई मात्रा में पानी पीते रहें।"}
    w_en = f" especially between {_fmt(window[0])} and {_fmt(window[1])}" if window else ""
    w_hi = f", ख़ासकर {_fmt(window[0])} से {_fmt(window[1])} के बीच" if window else ""
    lvl_en = {"high": "HIGH", "moderate": "moderate"}[level]
    lvl_hi = {"high": "ज़्यादा", "moderate": "मध्यम"}[level]
    do_en, do_hi = [], []
    if row.get("outdoor_routine", 0):
        do_en.append("postpone outdoor work or errands to the early morning or evening")
        do_hi.append("बाहर के काम सुबह जल्दी या शाम को करें")
    do_en.append("rest indoors in the coolest room" + ("" if row.get("home_ac", 0) else ", with a fan and a wet cloth"))
    do_hi.append("घर के सबसे ठंडे कमरे में आराम करें" + ("" if row.get("home_ac", 0) else ", पंखा और गीला कपड़ा रखें"))
    do_en.append("drink water regularly in the amount the doctor has advised")
    do_hi.append("डॉक्टर की बताई मात्रा में थोड़ा-थोड़ा पानी पीते रहें")
    en = (f"Namaste. HEATSHIELD heat alert for {first} ji: risk of heat strain on {date} is {lvl_en}{w_en}. "
          f"Please {'; '.join(do_en)}. Do not stop or change any medicine without asking the doctor. "
          f"If there is dizziness, confusion, fainting, very hot skin or vomiting, call the clinic or 108 at once.")
    hi = (f"नमस्ते। {first} जी के लिए HEATSHIELD गर्मी चेतावनी: {date} को गर्मी से तबीयत बिगड़ने का खतरा {lvl_hi} है{w_hi}। "
          f"कृपया {'; '.join(do_hi)}। डॉक्टर से पूछे बिना कोई दवा बंद या कम न करें। "
          f"चक्कर, उलझन, बेहोशी, बहुत गर्म त्वचा या उल्टी हो तो तुरंत क्लिनिक या 108 पर फ़ोन करें।")
    return {"level": level, "en": en, "hi": hi}
