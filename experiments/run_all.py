"""Reproduce everything: data -> world -> twin -> experiments E1-E4 -> results/.

    python experiments/run_all.py            # full run (~5 min on a laptop)
    python experiments/run_all.py --quick    # 60 patients, for a smoke test

E5-E7 (stress test, messy data, alerts) live in experiments/robustness.py; scenario checks in tests/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heatshield import config, ehr, weather                       # noqa: E402
from heatshield.config import cfg, PROCESSED, RESULTS, RAW_EHR, RAW_WEATHER, steps_per_day   # noqa: E402
from heatshield.world import simulate                               # noqa: E402
from heatshield.twin import filter as F, model                      # noqa: E402
from heatshield.twin.scenarios import build_horizon                 # noqa: E402
from heatshield.risk import features, train                         # noqa: E402

plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
COL = {"buller": "#9aa0a6", "population": "#c58b3c", "ehr": "#5b8def", "twin": "#d1495b", "truth": "#222222"}
NAMES = {"buller": "Standard HR method (Buller 2013)", "population": "Population model (no EHR)",
         "ehr": "EHR-aware, no learning", "twin": "HEATSHIELD twin (EHR + learning)"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def consts_for(mode: str, cohort: pd.DataFrame, k: model.Consts) -> model.Consts:
    if mode == "population":
        k.act_mult[:] = 1.0
        k.med_extra[:] = 0.04
    return k


def spread_for(mode: str, cohort: pd.DataFrame) -> dict | None:
    if mode == "twin":
        return F.unlearned_spread(cohort)
    if mode == "ehr":
        pri = model.ehr_priors(cohort)
        return {p: pri[p][1] for p in model.PARAMS if p != "hr_rest"}
    return None


def main(quick: bool = False):
    t_start = time.time()
    # ------------------------------------------------------------------ data
    if not (RAW_WEATHER / "Jaipur.csv").exists():
        weather.generate_synthetic()
    log(f"weather: {weather.weather_source()}")
    if not (RAW_EHR / "patients.csv").exists():
        ehr.generate_synthea_style()
    cohort = ehr.load_cohort()
    if quick:
        cohort = cohort.iloc[:60].reset_index(drop=True)
    simulate.run(cohort)
    D = dict(np.load(PROCESSED / "world.npz"))
    traits = pd.read_parquet(PROCESSED / "hidden_traits.parquet")
    spd = steps_per_day()
    cal_days = cfg()["simulation"]["calibration_days"]
    t_hide = cal_days * spd - 1
    evenings = features.evening_steps()
    bb = cohort["beta_blocker"].to_numpy() == 1
    log(f"cohort {len(cohort)} patients, {bb.mean():.0%} beta-blocked")

    # ------------------------------------------------------------------ filters
    est = {"buller": F.buller_filter(D["hr_obs"])}
    posts = {}
    for mode in ("population", "ehr", "twin"):
        log(f"filtering: {mode}")
        posts[mode] = F.run_filter(cohort, D, mode, snapshot_steps=set(evenings) | {t_hide})
        est[mode] = posts[mode]["tc_mean"]

    # ------------------------------------------------------------------ E1 core temperature
    tc = D["true_tc"]
    hot = tc >= 37.8
    rows = []
    for m, e in est.items():
        err = (e - tc) ** 2
        for g, gm in (("beta-blocked", bb), ("not beta-blocked", ~bb)):
            rows.append({"method": NAMES[m], "group": g,
                         "rmse_all_c": np.sqrt(err[gm].mean()),
                         "rmse_heat_strain_c": np.sqrt(err[gm][hot[gm]].mean()),
                         "bias_heat_strain_c": (e - tc)[gm][hot[gm]].mean()})
    e1 = pd.DataFrame(rows)
    e1.to_csv(RESULTS / "e1_core_temperature.csv", index=False)
    log("E1 done")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    piv = e1.pivot(index="method", columns="group", values="rmse_heat_strain_c").reindex([NAMES[m] for m in est])
    x = np.arange(len(piv))
    ax.bar(x - 0.2, piv["not beta-blocked"], 0.38, label="Not beta-blocked", color="#9fb7d9")
    ax.bar(x + 0.2, piv["beta-blocked"], 0.38, label="Beta-blocked", color="#d1495b")
    for i, (a, b) in enumerate(zip(piv["not beta-blocked"], piv["beta-blocked"])):
        ax.text(i - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [n.replace(" (", "\n(") for n in piv.index], fontsize=8)
    ax.set_ylabel("Core-temp error during heat strain (RMSE, °C)")
    ax.set_title("Standard HR method fails beta-blocked patients; the medication-aware twin does not", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "e1_core_temperature.png")
    plt.close(fig)

    # example trace: a beta-blocked patient on a hot outdoor day
    cand = np.where(bb & (cohort["outdoor_routine"].to_numpy() == 1))[0]
    i = cand[np.argmax(tc[cand].max(1))] if len(cand) else 0
    d0 = int(np.argmax(tc[i].reshape(-1, spd).max(1)))
    sl = slice(d0 * spd, (d0 + 1) * spd)
    hrs = np.arange(spd) * cfg()["simulation"]["step_minutes"] / 60
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(hrs, tc[i, sl], color=COL["truth"], lw=2, label="True core temp (hidden)")
    ax.plot(hrs, est["buller"][i, sl], color=COL["buller"], lw=1.5, label=NAMES["buller"])
    ax.plot(hrs, est["twin"][i, sl], color=COL["twin"], lw=1.5, ls="--", label=NAMES["twin"])
    ax.fill_between(hrs, est["twin"][i, sl] - 1.645 * posts["twin"]["tc_sd"][i, sl],
                    est["twin"][i, sl] + 1.645 * posts["twin"]["tc_sd"][i, sl], color=COL["twin"], alpha=.15)
    ax.set_xlabel("Hour of day"), ax.set_ylabel("Core temperature (°C)")
    ax.set_title(f"{cohort.loc[i, 'pid']}: beta-blocked, outdoor afternoon. HR alone misses the rise", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "e1_example_trace.png")
    plt.close(fig)

    # ------------------------------------------------------------------ E2 trait learning
    rows = []
    for p in model.LEARNED if hasattr(model, "LEARNED") else cfg()["twin"]["learned_params"]:
        truth = traits[p].to_numpy()
        tw = posts["twin"]["params"][p]
        rows.append({"trait": p, "population_mae": np.abs(model.population_point(cohort)[p] - truth).mean(),
                     "ehr_prior_mae": np.abs(tw[:, 0] - truth).mean(),
                     "after_day1_mae": np.abs(tw[:, spd - 1] - truth).mean(),
                     "after_day5_mae": np.abs(tw[:, t_hide] - truth).mean(),
                     "corr_day5": np.corrcoef(tw[:, t_hide], truth)[0, 1]})
    e2 = pd.DataFrame(rows)
    e2.to_csv(RESULTS / "e2_trait_learning.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    days = np.arange(tc.shape[1]) / spd
    truth = traits["hr_gain"].to_numpy()
    err = np.abs(posts["twin"]["params"]["hr_gain"] - truth[:, None])
    for g, gm, c in (("Beta-blocked", bb, "#d1495b"), ("Not beta-blocked", ~bb, "#5b8def")):
        ax.plot(days, err[gm].mean(0), color=c, label=g)
    ax.set_xlabel("Days of wearable data"), ax.set_ylabel("HR-response error (bpm/°C)")
    ax.set_title("The twin learns each patient's heart-rate response to heat", fontsize=10)
    ax.axvline(cal_days, color="#888", ls=":", lw=1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "e2_trait_learning.png")
    plt.close(fig)
    log("E2 done")

    # ------------------------------------------------------------------ E3 hide-and-reveal
    h = (cfg()["simulation"]["days"] - cal_days) * spd - 1
    h = min(h, 2 * spd)
    rows, fc_store = [], {}
    for mode in ("population", "ehr", "twin"):
        H, k, _ = build_horizon(cohort, D, t_hide, h, actual=True)
        res = F.forecast(posts[mode]["snapshots"][t_hide], consts_for(mode, cohort, k), H, seed=3,
                         spread=spread_for(mode, cohort))
        fc_store[mode] = res
        true = tc[:, t_hide + 1:t_hide + 1 + h]
        true_psi = D["true_psi"][:, t_hide + 1:t_hide + 1 + h]
        inside = (true >= res["tc_lo"]) & (true <= res["tc_hi"])
        for g, gm in (("all", np.ones(len(cohort), bool)), ("beta-blocked", bb), ("not beta-blocked", ~bb)):
            rows.append({"model": NAMES[mode], "group": g,
                         "tc_rmse_c": np.sqrt(((res["tc_mid"] - true) ** 2)[gm].mean()),
                         "daily_peak_tc_mae_c": np.abs(res["tc_mid"][gm].reshape(gm.sum(), -1, spd)[:, :, :].max(2)
                                                       - true[gm].reshape(gm.sum(), -1, spd).max(2)).mean()
                         if h % spd == 0 else np.nan,
                         "peak_psi_mae": np.abs(res["psi_mid"][gm].max(1) - true_psi[gm].max(1)).mean(),
                         "band_90_coverage": inside[gm].mean()})
    e3 = pd.DataFrame(rows)
    e3.to_csv(RESULTS / "e3_hide_and_reveal.csv", index=False)

    j = i
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    hh = np.arange(1, h + 1) * cfg()["simulation"]["step_minutes"] / 60
    r = fc_store["twin"]
    ax.fill_between(hh, r["tc_lo"][j], r["tc_hi"][j], color=COL["twin"], alpha=.18, label="Twin 90% band")
    ax.plot(hh, r["tc_mid"][j], color=COL["twin"], lw=1.6, ls="--", label="Twin forecast (no sensor data)")
    ax.plot(hh, tc[j, t_hide + 1:t_hide + 1 + h], color=COL["truth"], lw=1.8, label="Revealed truth")
    ax.set_xlabel("Hours after the sensor data stops (end of day 5)"), ax.set_ylabel("Core temperature (°C)")
    ax.set_title("Hide-and-reveal: the twin forecasts two days it never saw", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "e3_hide_and_reveal.png")
    plt.close(fig)
    log("E3 done")

    # ------------------------------------------------------------------ E4 risk models
    twin_fc = {}
    tp = posts["twin"]
    for t0 in evenings:
        H, k, _ = build_horizon(cohort, D, t0, spd, actual=False)
        res = F.forecast(tp["snapshots"][t0], k, H, seed=t0, spread=F.unlearned_spread(cohort))
        twin_fc[t0] = {"tc_now": tp["tc_mean"][:, t0], "def_now": tp["def_mean"][:, t0],
                       "hr_gain": tp["params"]["hr_gain"][:, t0], "hr_rest": tp["params"]["hr_rest"][:, t0],
                       "p_event": res["p_event"], "peak_psi": res["peak_psi"], "peak_tc": res["peak_tc"]}
    feats = features.build(cohort, D, twin_fc)
    feats.to_parquet(PROCESSED / "features.parquet", index=False)
    e4, preds = train.train_and_evaluate(feats)
    e4.to_csv(RESULTS / "e4_risk_models.csv", index=False)
    e4m = train.multi_split(feats, 5)
    e4m.to_csv(RESULTS / "e4_risk_models_5splits.csv", index=False)
    preds.to_parquet(PROCESSED / "predictions.parquet", index=False)
    log(f"E4 done: event rate {feats['label'].mean():.1%} over {len(feats)} patient-evenings")

    test = preds[preds["split"] == "test"]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.5))
    from sklearn.metrics import roc_curve
    for name, c in (("heat_index_rule", "#9aa0a6"), ("xgb_no_twin", "#5b8def"), ("twin_only", "#c58b3c"),
                    ("xgb_with_twin", "#d1495b")):
        fpr, tpr, _ = roc_curve(test["label"], test[name])
        au = e4[(e4.model == train.LABELS[name]) & (e4.group == "all")]["auroc"].iloc[0]
        axs[0].plot(fpr, tpr, color=c, label=f"{train.LABELS[name]} ({au:.2f})")
        bins = np.linspace(0, 1, 6)
        idx = np.clip(np.digitize(test[name], bins) - 1, 0, 4)
        mp = [test[name][idx == b].mean() for b in range(5) if (idx == b).sum() >= 5]
        fo = [test["label"][idx == b].mean() for b in range(5) if (idx == b).sum() >= 5]
        if name != "heat_index_rule":
            axs[1].plot(mp, fo, "o-", color=c, ms=4)
    axs[0].plot([0, 1], [0, 1], color="#ccc", lw=.8)
    axs[0].set_xlabel("False positive rate"), axs[0].set_ylabel("True positive rate")
    axs[0].set_title("Discrimination (AUROC)", fontsize=10), axs[0].legend(frameon=False, fontsize=7)
    axs[1].plot([0, 1], [0, 1], color="#ccc", lw=.8)
    axs[1].set_xlabel("Predicted risk"), axs[1].set_ylabel("Observed event rate")
    axs[1].set_title("Calibration (test patients)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "e4_risk_models.png")
    plt.close(fig)

    # ------------------------------------------------------------------ summary
    summary = {"weather_source": weather.weather_source(), "n_patients": len(cohort),
               "patient_evenings": len(feats), "event_rate": float(feats["label"].mean()),
               "runtime_s": round(time.time() - t_start, 1)}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        for name, df in (("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4[e4.group.isin(["all", "beta-blocked"])]),
                         ("E4 over 5 patient splits", e4m[e4m.group.isin(["all", "beta-blocked"])][
                             ["model", "group", "auroc_mean", "auroc_sd", "auprc_mean", "brier_mean"]])):
            print(f"\n=== {name} ===\n{df.to_string(index=False)}")
    log(f"all done in {summary['runtime_s']} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
