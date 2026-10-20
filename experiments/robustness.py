"""Robustness and clinical-usefulness experiments (run after run_all.py).

    python experiments/robustness.py           # ~15 min on a laptop
    python experiments/robustness.py --quick   # 60 patients, smoke test

E5  Stress test. The world simulator's medication effects are changed (weaker, stronger, absent)
    while the twin keeps its ORIGINAL assumptions. Answers: "you built the beta-blocker effect
    into your own simulator, so of course the twin wins."
E6  Messy wearable data. Charging gaps, nights without the watch, random dropouts and motion
    spikes are added to the wrist data the twin sees. The hidden truth is unchanged.
E7  Warning time and alert burden. Each model's alert threshold is set on validation patients to
    catch at least 80% of events; on test patients we report events caught, alerts per patient
    per week, false alarms, and how many hours ahead of the danger the alert arrives.

Nothing here overwrites the main dataset, models or results of run_all.py.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heatshield import ehr                                            # noqa: E402
from heatshield.config import cfg, RESULTS, steps_per_day             # noqa: E402
from heatshield.psi import is_event                                   # noqa: E402
from heatshield.world import simulate                                 # noqa: E402
from heatshield.twin import filter as F                               # noqa: E402
from heatshield.twin.scenarios import build_horizon                   # noqa: E402
from heatshield.risk import features, train                           # noqa: E402

plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
MODEL_COLS = {"heat_index_rule": "#9aa0a6", "xgb_no_twin": "#5b8def", "twin_only": "#c58b3c",
              "xgb_with_twin": "#d1495b"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- shared evaluation
def evaluate_world(cohort, D, n_splits=5, keep=False):
    """Run Buller + the three filters on one world, return headline metrics (and optionally artefacts)."""
    spd = steps_per_day()
    t_hide = cfg()["simulation"]["calibration_days"] * spd - 1
    evenings = features.evening_steps()
    bb = cohort["beta_blocker"].to_numpy() == 1
    tc = D["true_tc"]
    hot = tc >= 37.8

    est = {"buller": F.buller_filter(D["hr_obs"])}
    posts = {}
    for mode in ("population", "ehr", "twin"):
        posts[mode] = F.run_filter(cohort, D, mode, snapshot_steps=set(evenings) | {t_hide})
        est[mode] = posts[mode]["tc_mean"]
    out = {}
    for m, e in est.items():
        out[f"rmse_bb_{m}"] = float(np.sqrt(((e - tc) ** 2)[bb[:, None] & hot].mean()))
        out[f"rmse_nbb_{m}"] = float(np.sqrt(((e - tc) ** 2)[~bb[:, None] & hot].mean()))

    # hide-and-reveal band for the twin (2 days without sensor data)
    h = 2 * spd
    H, k, _ = build_horizon(cohort, D, t_hide, h, actual=True)
    r = F.forecast(posts["twin"]["snapshots"][t_hide], k, H, seed=3, spread=F.unlearned_spread(cohort))
    true = tc[:, t_hide + 1:t_hide + 1 + h]
    out["twin_band90_coverage"] = float(((true >= r["tc_lo"]) & (true <= r["tc_hi"])).mean())
    out["twin_band90_width_c"] = float((r["tc_hi"] - r["tc_lo"]).mean())
    out["twin_live_tc_sd_c"] = float(posts["twin"]["tc_sd"].mean())

    # evening risk forecasts -> risk models over several patient splits
    tp = posts["twin"]
    twin_fc = {}
    for t0 in evenings:
        H, k, _ = build_horizon(cohort, D, t0, spd, actual=False)
        res = F.forecast(tp["snapshots"][t0], k, H, seed=t0, spread=F.unlearned_spread(cohort))
        twin_fc[t0] = {"tc_now": tp["tc_mean"][:, t0], "def_now": tp["def_mean"][:, t0],
                       "hr_gain": tp["params"]["hr_gain"][:, t0], "hr_rest": tp["params"]["hr_rest"][:, t0],
                       "p_event": res["p_event"], "peak_psi": res["peak_psi"], "peak_tc": res["peak_tc"]}
    feats = features.build(cohort, D, twin_fc)
    ms = train.multi_split(feats, n_splits)
    allg = ms[ms.group == "all"].set_index("model")
    for name, label in train.LABELS.items():
        out[f"auroc_{name}"] = float(allg.loc[label, "auroc_mean"])
        out[f"auprc_{name}"] = float(allg.loc[label, "auprc_mean"])
    out["event_rate"] = float(feats["label"].mean())
    if keep:
        return out, feats
    return out


# ----------------------------------------------------------------------------- E5 stress test
def world_variants():
    """Each variant rescales the world's medication EFFECT (distance from 'no effect')."""
    med0 = copy.deepcopy(cfg()["world"]["medication"])

    def scaled(bb_scale, fluid_scale):
        m = copy.deepcopy(med0)
        m["beta_blocker_hr_gain_mult"] = 1 - (1 - med0["beta_blocker_hr_gain_mult"]) * bb_scale
        m["beta_blocker_activity_mult"] = 1 - (1 - med0["beta_blocker_activity_mult"]) * bb_scale
        m["beta_blocker_rest_hr_shift"] = med0["beta_blocker_rest_hr_shift"] * bb_scale
        m["diuretic_extra_l_per_h"] = med0["diuretic_extra_l_per_h"] * fluid_scale
        m["sglt2i_extra_l_per_h"] = med0["sglt2i_extra_l_per_h"] * fluid_scale
        return m

    return med0, [
        ("As assumed (baseline)", scaled(1.0, 1.0)),
        ("Beta-blocker effect 50% weaker", scaled(0.5, 1.0)),
        ("Beta-blocker effect 50% stronger", scaled(1.5, 1.0)),
        ("Fluid-losing drugs 50% weaker", scaled(1.0, 0.5)),
        ("Fluid-losing drugs 50% stronger", scaled(1.0, 1.5)),
        ("All wrong at once (BB weaker, fluids stronger)", scaled(0.5, 1.5)),
        ("Drugs have NO effect at all", scaled(0.0, 0.0)),
    ]


def e5_stress(cohort, n_splits):
    med0, variants = world_variants()
    rows = []
    try:
        for name, med in variants:
            log(f"E5 world: {name}")
            cfg()["world"]["medication"] = med
            D, _ = simulate.run(cohort, save=False)
            rows.append({"world": name, **evaluate_world(cohort, D, n_splits)})
    finally:
        cfg()["world"]["medication"] = med0
    e5 = pd.DataFrame(rows)
    e5.to_csv(RESULTS / "e5_stress_test.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(e5))[::-1]
    ax = axs[0]
    for m, c, lab in (("buller", "#9aa0a6", "Standard HR method"), ("population", "#c58b3c", "Population model"),
                      ("twin", "#d1495b", "HEATSHIELD twin")):
        ax.plot(e5[f"rmse_bb_{m}"], y, "o", color=c, label=lab, ms=7)
    ax.set_yticks(y, e5["world"], fontsize=8)
    ax.set_xlabel("Core-temp error, beta-blocked, during heat strain (°C)")
    ax.set_title("Core temperature", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axs[1]
    for m, c in MODEL_COLS.items():
        ax.plot(e5[f"auroc_{m}"], y, "o", color=c, label=train.LABELS[m], ms=7)
    ax.set_yticks(y, [""] * len(y))
    ax.set_xlabel("AUROC, next-24-h adverse event (mean of 5 splits)")
    ax.set_title("Event prediction", fontsize=10)
    ax.set_xlim(0.4, 1.0)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Stress test: the world's drug effects change, the twin's assumptions do not", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "e5_stress_test.png")
    plt.close(fig)
    return e5


# ----------------------------------------------------------------------------- E6 messy wearable
def degrade(D, level, rng):
    """Return a copy of D with realistic wearable problems added to hr_obs / skin_obs only."""
    spec = {"clean (as simulated)": None,
            "realistic": dict(charge_h=1.5, night_off=0.25, rand=0.10, spike=0.03),
            "poor": dict(charge_h=3.0, night_off=0.5, rand=0.25, spike=0.08)}[level]
    D2 = dict(D)
    hr, sk = D["hr_obs"].copy(), D["skin_obs"].copy()
    if spec:
        P, T = hr.shape
        spd = steps_per_day()
        step_h = cfg()["simulation"]["step_minutes"] / 60
        days = T // spd
        miss = rng.random((P, T)) < spec["rand"]
        n_charge = int(round(spec["charge_h"] / step_h))
        for d in range(days):
            # charging: one block per day, starting somewhere between 07:00 and 21:00
            start = d * spd + rng.integers(int(7 / step_h), int(21 / step_h), P)
            for i in range(P):
                miss[i, start[i]:start[i] + n_charge] = True
            # some nights the watch is left on the table (22:00-06:00)
            off = rng.random(P) < spec["night_off"]
            a, b = d * spd + int(22 / step_h), min(d * spd + int(30 / step_h), T)
            miss[off, a:b] = True
        spikes = (rng.random((P, T)) < spec["spike"]) & ~miss
        hr = hr + spikes * rng.choice([-1, 1], (P, T)) * rng.uniform(15, 35, (P, T))
        hr[miss], sk[miss] = np.nan, np.nan
    D2["hr_obs"], D2["skin_obs"] = hr, sk
    return D2, float(np.isnan(hr).mean())


def e6_messy(cohort, D, n_splits):
    rows = []
    for level in ("clean (as simulated)", "realistic", "poor"):
        log(f"E6 wearable data: {level}")
        D2, frac = degrade(D, level, np.random.default_rng(7))
        rows.append({"wearable_data": level, "missing_fraction": frac, **evaluate_world(cohort, D2, n_splits)})
    e6 = pd.DataFrame(rows)
    e6.to_csv(RESULTS / "e6_messy_wearable.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
    x = np.arange(len(e6))
    labels = [f"{w}\n{m:.0%} missing" for w, m in zip(e6["wearable_data"], e6["missing_fraction"])]
    ax = axs[0]
    ax.plot(x, e6["rmse_bb_buller"], "o-", color="#9aa0a6", label="Standard HR method")
    ax.plot(x, e6["rmse_bb_twin"], "o-", color="#d1495b", label="HEATSHIELD twin")
    ax.set_ylabel("°C"), ax.set_title("Core-temp error, beta-blocked", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axs[1]
    ax.bar(x, e6["twin_band90_coverage"], color="#d1495b", alpha=.8)
    for xi, c, w in zip(x, e6["twin_band90_coverage"], e6["twin_band90_width_c"]):
        ax.text(xi, c + 0.01, f"{c:.0%}\nwidth {w:.2f} °C", ha="center", fontsize=7)
    ax.set_ylim(0, 1.1), ax.set_title("Twin 90% band: coverage", fontsize=9)
    ax = axs[2]
    for m, c in MODEL_COLS.items():
        ax.plot(x, e6[f"auroc_{m}"], "o-", color=c, label=train.LABELS[m])
    ax.set_ylim(0.4, 1.0), ax.set_title("Event AUROC (5 splits)", fontsize=9)
    ax.legend(frameon=False, fontsize=6)
    for ax in axs:
        ax.set_xticks(x, labels, fontsize=7)
    fig.suptitle("Messy wearable data: charging gaps, nights off-wrist, dropouts, motion spikes", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "e6_messy_wearable.png")
    plt.close(fig)
    return e6


# ----------------------------------------------------------------------------- E7 alerts
def first_event_hours(D, feats):
    """Hours from the 20:00 forecast to the first moment of danger (NaN if no event)."""
    spd = steps_per_day()
    step_h = cfg()["simulation"]["step_minutes"] / 60
    th = cfg()["outcome"]
    pid_idx = {p: i for i, p in enumerate(sorted(feats["pid"].unique()))}
    out = np.full(len(feats), np.nan)
    for r, (pid, t0) in enumerate(zip(feats["pid"], feats["t0"])):
        i = pid_idx[pid]
        sl = slice(t0 + 1, t0 + 1 + spd)
        hit = (D["true_psi"][i, sl] >= th["psi_threshold"]) | (D["true_tc"][i, sl] >= th["core_threshold_c"])
        if hit.any():
            out[r] = (np.argmax(hit) + 1) * step_h
    return out


def e7_alerts(D, feats, n_splits, target_sens=0.8):
    lead = first_event_hours(D, feats)
    rows, leads = [], {m: [] for m in MODEL_COLS}
    for k in range(n_splits):
        _, preds = train.train_and_evaluate(feats, seed=cfg()["seed"] + k, save=False)
        preds["lead_h"] = lead
        val, test = preds[preds.split == "val"], preds[preds.split == "test"]
        for m in MODEL_COLS:
            # highest threshold that still catches >= target_sens of validation events
            pos = np.sort(val.loc[val.label == 1, m].to_numpy())
            thr = pos[int(np.floor((1 - target_sens) * len(pos)))] if len(pos) else 0.5
            alert = test[m].to_numpy() >= thr
            y = test["label"].to_numpy() == 1
            n_pw = test["pid"].nunique() * (test["evening"].nunique() / 7)    # patient-weeks
            rows.append({"model": train.LABELS[m], "split": k, "threshold": thr,
                         "sensitivity": (alert & y).sum() / max(y.sum(), 1),
                         "ppv": (alert & y).sum() / max(alert.sum(), 1),
                         "alerts_per_patient_week": alert.sum() / n_pw,
                         "false_alarms_per_patient_week": (alert & ~y).sum() / n_pw,
                         "missed_events": int((~alert & y).sum()), "events": int(y.sum())})
            leads[m] += list(test.loc[alert & y, "lead_h"])
    allr = pd.DataFrame(rows)
    e7 = allr.groupby("model", sort=False).mean(numeric_only=True).drop(columns=["split"]).reset_index()
    e7["median_warning_h"] = [np.median(leads[m]) if leads[m] else np.nan for m in MODEL_COLS]
    e7["warning_h_p10"] = [np.percentile(leads[m], 10) if leads[m] else np.nan for m in MODEL_COLS]
    e7.to_csv(RESULTS / "e7_alerts.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axs[0]
    x = np.arange(len(e7))
    ax.bar(x - 0.2, e7["alerts_per_patient_week"], 0.38, color="#bbb", label="All alerts")
    ax.bar(x + 0.2, e7["false_alarms_per_patient_week"], 0.38, color="#d1495b", label="False alarms")
    for xi, a, s in zip(x, e7["alerts_per_patient_week"], e7["sensitivity"]):
        ax.text(xi - 0.2, a + 0.03, f"catches {s:.0%}", ha="center", fontsize=7)
    ax.set_xticks(x, [m.replace(" (", "\n(") for m in e7["model"]], fontsize=7)
    ax.set_ylabel("per patient per week")
    ax.set_title(f"Alert burden at ≥{target_sens:.0%} sensitivity (set on validation patients)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    lt = [h for h in leads["xgb_with_twin"]]
    ax.hist(lt, bins=np.arange(0, 25, 1), color="#d1495b", alpha=.85)
    ax.axvline(np.median(lt), color="#222", ls="--", lw=1)
    ax.text(np.median(lt) + 0.3, ax.get_ylim()[1] * 0.9, f"median {np.median(lt):.0f} h", fontsize=8)
    ax.set_xlabel("Hours between the 8 pm alert and the first moment of danger")
    ax.set_ylabel("caught events"), ax.set_title("Warning time (HEATSHIELD)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "e7_alerts.png")
    plt.close(fig)
    return e7


# ----------------------------------------------------------------------------- main
def main(quick=False):
    t0 = time.time()
    cohort = ehr.load_cohort()
    if quick:
        cohort = cohort.iloc[:60].reset_index(drop=True)
    n_splits = 5
    e5 = e5_stress(cohort, n_splits)
    D, _ = simulate.run(cohort, save=False)          # baseline world, identical to run_all.py
    e6 = e6_messy(cohort, D, n_splits)
    log("E7 alerts")
    _, feats = evaluate_world(cohort, D, n_splits, keep=True)
    e7 = e7_alerts(D, feats, n_splits)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print("\n=== E5 stress test ===")
        print(e5[["world", "rmse_bb_buller", "rmse_bb_population", "rmse_bb_ehr", "rmse_bb_twin",
                  "auroc_heat_index_rule", "auroc_xgb_no_twin", "auroc_twin_only", "auroc_xgb_with_twin",
                  "event_rate"]].to_string(index=False))
        print("\n=== E6 messy wearable data ===")
        print(e6[["wearable_data", "missing_fraction", "rmse_bb_buller", "rmse_bb_twin", "twin_band90_coverage",
                  "twin_band90_width_c", "auroc_xgb_no_twin", "auroc_twin_only", "auroc_xgb_with_twin"]]
              .to_string(index=False))
        print("\n=== E7 alerts (>= 80% sensitivity on validation) ===")
        print(e7.to_string(index=False))
    log(f"robustness done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
