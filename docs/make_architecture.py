"""Draws docs/architecture.png and docs/architecture.pdf."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
fig, ax = plt.subplots(figsize=(12, 7.6))
ax.set_xlim(0, 120); ax.set_ylim(0, 76); ax.axis("off")
INK, QUIET, ACC, GRID = "#222", "#666", "#d1495b", "#bbb"


def box(x, y, w, h, title, lines=(), fc="white", ec=INK, ls="-", lw=1.2, tsize=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, ls=ls, lw=lw))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top", fontsize=tsize, weight="bold", color=INK)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 5.0 - 2.4 * i, l, ha="center", va="top", fontsize=8.6, color=QUIET)


def arrow(x1, y1, x2, y2, ls="-", label=None, lx=None, ly=None, color=INK):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, ls=ls))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2 + 0.8, ly if ly is not None else (y1 + y2) / 2, label,
                fontsize=8.2, color=QUIET, style="italic")


ax.text(1, 75, "HEATSHIELD architecture: the twin only sees what a real system would; hidden truth only grades it",
        fontsize=12.5, weight="bold", va="top")

# 1 world
ax.add_patch(FancyBboxPatch((1, 43), 118, 27, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#f3f3f3", ec=GRID))
ax.text(3, 68.5, "1 · World simulator (the fake reality, src/heatshield/world)", fontsize=10.5, weight="bold", va="top")
box(4, 55, 32, 10, "Synthetic EHR", ["Synthea-format CSVs: age, conditions,", "medications, eGFR, weight, AC"])
box(44, 55, 32, 10, "Real weather", ["Open-Meteo, May–June 2024 heatwave,", "5 Indian cities (hourly)"])
box(84, 55, 32, 10, "Hidden patient traits", ["sweat capacity, HR response,", "fluid loss, fluid reserve, baselines"])
box(30, 44.5, 60, 8, "Physiology engine (every 5 min)", ["heat + fluid balance, circadian rhythm, routines, sensor noise"])
arrow(20, 55, 30, 50); arrow(60, 55, 60, 52.5); arrow(100, 55, 90, 50)

box(10, 29.5, 36, 10, "Wearable stream (observed)", ["HR, skin temp, steps, sleep,", "phone location indoor/outdoor"])
box(80, 29.5, 36, 10, "Ground truth (hidden)", ["true core temp, fluid deficit,", "PSI, event labels"], ec=QUIET, ls="--")
arrow(40, 44.5, 30, 39, label="observed", lx=36, ly=41.5)
arrow(80, 44.5, 95, 39, ls="--", label="never shown to the twin", lx=88, ly=41.5, color=QUIET)

# 2 twin
ax.add_patch(FancyBboxPatch((1, 12), 78, 15, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#fbe9ec", ec=ACC, lw=1.8))
ax.text(3, 26, "2 · The twin, one per patient", fontsize=10.5, weight="bold", va="top", color=INK)
box(3, 13, 23, 9.5, "EHR priors", ["medication list + eGFR", "set starting beliefs"], tsize=10)
box(29, 13, 23, 9.5, "Particle filter", ["tracks core temp, learns", "HR response + resting HR"], tsize=10)
box(55, 13, 22, 9.5, "24 h forecast", ["PSI + 90% band under", "weather and what-ifs"], tsize=10)
arrow(26, 17.5, 29, 17.5); arrow(52, 17.5, 55, 17.5)
arrow(40.5, 29.5, 40.5, 22.8, label="+ weather forecast", lx=41.5, ly=25.2)
box(84, 13, 32, 10, "Evaluation E1–E7", ["graded against hidden truth; stress tests,", "messy data, alert burden; pytest"], ec=QUIET, ls="--")
arrow(98, 29.5, 98, 23.3, ls="--", color=QUIET)

# 3, 4
box(3, 0.5, 34, 9, "3 · Risk model", ["XGBoost benchmark, Platt-calibrated"])
box(44, 0.5, 34, 9, "4 · Clinician dashboard", ["triage, forecast, what-if, live replay,", "district view for health officers"])
arrow(14, 12.6, 14, 9.9, label="twin features", lx=15, ly=10.9)
arrow(66, 12.6, 61, 9.9, label="forecast, scenarios, drivers", lx=65, ly=10.9)
arrow(37, 4.7, 44, 4.7)

box(84, 0.5, 32, 9, "5 · Outputs", ["family alert draft (Hindi / English),", "doctor-approved · FHIR R4 export"], fc="#fbe9ec", ec=ACC)
arrow(78, 4.7, 84, 4.7)
fig.tight_layout()
fig.savefig(OUT / "architecture.png", dpi=170)
fig.savefig(OUT / "architecture.pdf")
print("saved")
