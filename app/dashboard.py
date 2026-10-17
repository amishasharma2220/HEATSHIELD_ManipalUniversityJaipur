"""HEATSHIELD clinician dashboard.   Run:  streamlit run app/dashboard.py

Line conventions everywhere: solid = observed, dashed = estimated, shaded = uncertainty,
dotted = hypothetical scenario.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heatshield.config import PROCESSED, RESULTS, CONFIG_PATH, cfg, steps_per_day   # noqa: E402
from heatshield.twin.twin import PatientTwin                                       # noqa: E402
from heatshield.twin.scenarios import Scenario, PRESETS                            # noqa: E402
from heatshield.weather import weather_source                                      # noqa: E402

st.set_page_config(page_title="HEATSHIELD", page_icon="🌡️", layout="wide")

RISK = "#d1495b"
EST = "#5b8def"
OBS = "#2d3436"
HYP = "#8e44ad"
QUIET = "#9aa0a6"
HEAT_DRUGS = {"beta_blocker": "Beta-blocker: blunts heart-rate rise, hides the usual warning sign",
              "diuretic": "Diuretic: extra fluid loss, dehydrates faster",
              "sglt2i": "SGLT2 inhibitor: glucose in urine pulls water with it",
              "acei_arb": "ACE inhibitor / ARB: less kidney protection when fluid is low"}
SPD = steps_per_day()
EVENING = cfg()["outcome"]["evening_hour"] * 60 // cfg()["simulation"]["step_minutes"]


# ------------------------------------------------------------------------------------ data
@st.cache_resource
def load():
    cohort = pd.read_parquet(PROCESSED / "cohort.parquet")
    D = dict(np.load(PROCESSED / "world.npz"))
    if len(D["hr_obs"]) != len(cohort):
        cohort = cohort.iloc[:len(D["hr_obs"])].reset_index(drop=True)
    preds = pd.read_parquet(PROCESSED / "predictions.parquet") if (PROCESSED / "predictions.parquet").exists() else None
    return cohort, D, preds


@st.cache_resource(max_entries=24)
def twin_at(pid: str, t: int) -> PatientTwin:
    cohort, D, _ = load()
    return PatientTwin(cohort, D, pid).update(t)


@st.cache_data(max_entries=200)
def forecast_cached(pid: str, t: int, sc: tuple):
    sc = Scenario(**dict(sc))
    return twin_at(pid, t).forecast(sc)


def fc(pid, t, sc: Scenario):
    return forecast_cached(pid, t, tuple(sorted(sc.as_dict().items())))


@st.cache_data(max_entries=50)
def same_weather_forecast(pid2: str, pid1: str, t: int):
    """Forecast patient 2 under patient 1's weather (same heatwave, different body)."""
    import copy
    cohort_, D_, _ = load()
    tw2 = copy.copy(twin_at(pid2, t))
    i1 = int(np.where(cohort_["pid"].to_numpy() == pid1)[0][0])
    tw2.D = dict(tw2.D)
    ac2 = bool(tw2.profile.loc[0, "home_ac"])
    for k in ("t_out", "t_out_forecast", "rh", "solar"):
        tw2.D[k] = D_[k][[i1]]
    from heatshield.weather import indoor_temp
    tw2.D["t_in"] = indoor_temp(D_["t_out"][[i1]], np.array([ac2]))
    return tw2.forecast(Scenario())


def band(p: float) -> str:
    return "High" if p >= 0.5 else "Moderate" if p >= 0.2 else "Low"


if not (PROCESSED / "world.npz").exists() or not (PROCESSED / "predictions.parquet").exists():
    st.title("🌡️ HEATSHIELD")
    st.info("No simulated data yet. Build it once (about 4 minutes): this generates the synthetic cohort, "
            "runs the world simulator and the twin, and trains the risk models. "
            "Same as running `python experiments/run_all.py`.")
    if st.button("Build data now", type="primary"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
        import run_all
        with st.spinner("Building... (about 4 minutes)"):
            run_all.main()
        st.cache_resource.clear()
        st.rerun()
    st.stop()

cohort, D, preds = load()
n_days = cfg()["simulation"]["days"]

# ------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🌡️ HEATSHIELD")
    st.caption("Medication-aware digital twin for heat strain in vulnerable patients")
    st.warning("Prototype on synthetic data. Not validated for clinical use.", icon="⚠️")
    day = st.slider("Replay the wearable stream up to the evening of day", 1, n_days - 1, n_days - 1,
                    help="The twin re-filters every reading up to 8 pm on this day, then forecasts the next 24 h.")
    t_now = (day - 1) * SPD + EVENING
    st.caption(f"Weather: {weather_source().split('(')[0].strip()}")
    st.caption("Lines: solid = observed · dashed = estimated · shaded = uncertainty · dotted = hypothetical")

# ------------------------------------------------------------------------------------ triage
st.title("Tomorrow's heat-strain risk")
if preds is not None:
    p = preds[preds["evening"] == day].set_index("pid")
    tri = cohort.set_index("pid").join(p[["xgb_with_twin", "twin_only"]], how="inner")
    tmax = pd.Series(D["t_out_forecast"][:, t_now + 1:t_now + 1 + SPD].max(1), index=cohort["pid"])
    tri["Tomorrow max °C"] = tmax.reindex(tri.index).round(1)
    tri["Risk"] = tri["twin_only"]                 # the twin's own forecast: needs no labelled outcomes
    tri["ML risk"] = tri["xgb_with_twin"]
    tri["Band"] = tri["Risk"].map(band)
    tri["Heat-relevant meds"] = tri.apply(lambda r: ", ".join(
        lbl for col, lbl in (("beta_blocker", "β-blocker"), ("diuretic", "diuretic"), ("sglt2i", "SGLT2i"))
        if r[col]) or "—", axis=1)
    tri["Home AC"] = tri["home_ac"].map({1: "yes", 0: "no"})
    show = tri.sort_values("Risk", ascending=False)[
        ["name", "age", "city", "Heat-relevant meds", "Home AC", "Tomorrow max °C", "Risk", "Band", "ML risk"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients monitored", len(show))
    c2.metric("High risk tomorrow", int((show["Band"] == "High").sum()))
    c3.metric("Moderate risk", int((show["Band"] == "Moderate").sum()))
    sel = st.dataframe(show.reset_index().rename(columns={"pid": "ID", "name": "Patient"}), hide_index=True,
                       height=260, width="stretch", on_select="rerun", selection_mode="single-row",
                       column_config={"Risk": st.column_config.ProgressColumn("Twin risk (24 h)", min_value=0, max_value=1,
                                                                              format="percent"),
                                      "ML risk": st.column_config.NumberColumn("XGBoost + twin", format="percent")})
    default = show.index[0]
    if sel and sel.selection.rows:
        default = show.index[sel.selection.rows[0]]
else:
    st.info("Run `python experiments/run_all.py` first to compute the triage list.")
    default = cohort["pid"].iloc[0]

pids = cohort["pid"].tolist()
pid = st.selectbox("Open patient", pids, index=pids.index(default),
                   format_func=lambda x: f"{x} · {cohort.set_index('pid').loc[x, 'name']}")
row = cohort.set_index("pid").loc[pid]
tw = twin_at(pid, t_now)
base = fc(pid, t_now, PRESETS["Baseline"])

tabs = st.tabs(["1 · Overview", "2 · Live twin state", "3 · 24 h forecast", "4 · What-if simulator",
                "5 · Evidence & limits"])

# ------------------------------------------------------------------------------------ 1 overview
with tabs[0]:
    a, b = st.columns([1, 1.3])
    with a:
        st.subheader(f"{row['name']}, {row['age']} {row['sex']}")
        conds = [n for c, n in (("hypertension", "Hypertension"), ("t2d", "Type 2 diabetes"), ("ckd", "CKD")) if row[c]]
        st.write("**Conditions:** " + (", ".join(conds) or "none recorded"))
        st.write(f"**eGFR:** {row['egfr']:.0f} mL/min" + ("  ⚠️ reduced" if row["egfr"] < 60 else ""))
        st.write(f"**Weight:** {row['weight_kg']:.0f} kg · **City:** {row['city']}")
        st.write(f"**Home AC:** {'yes' if row['home_ac'] else 'no'} · **Afternoon outdoor routine:** "
                 f"{'yes' if row['outdoor_routine'] else 'no'}")
    with b:
        st.subheader("Medications")
        for m in str(row["medications"]).split("; "):
            if m:
                st.write("• " + m)
        flags = [txt for col, txt in HEAT_DRUGS.items() if row[col]]
        if flags:
            st.markdown("**Heat-relevant:**")
            for f in flags:
                st.markdown(f"<span style='color:{RISK}'>▲</span> {f}", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------ 2 live state
with tabs[1]:
    h = tw.history()
    h = h[h["time"] >= h["time"].iloc[-1] - pd.Timedelta(hours=48)]
    s = tw.state()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated core temp", f"{s['tc']:.2f} °C", help="Estimated by the twin; never measured directly")
    m2.metric("Estimated fluid deficit", f"{s['deficit_l']:.2f} L")
    m3.metric("Learned HR response", f"{s['hr_gain']:.0f} ± {s['hr_gain_sd']:.1f} bpm/°C")
    m4.metric("Learned resting HR", f"{s['hr_rest']:.0f} ± {s['hr_rest_sd']:.1f} bpm")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Heart rate (observed)", "Core temperature (estimated) & skin (observed)",
                                        "Outdoor temperature (weather) and time outdoors"))
    fig.add_trace(go.Scatter(x=h.time, y=h.hr_obs, line=dict(color=OBS, width=1), name="HR"), 1, 1)
    up, lo = h.tc_est + 1.645 * h.tc_sd, h.tc_est - 1.645 * h.tc_sd
    fig.add_trace(go.Scatter(x=pd.concat([h.time, h.time[::-1]]), y=pd.concat([up, lo[::-1]]), fill="toself",
                             fillcolor="rgba(91,141,239,.18)", line=dict(width=0), name="90% band"), 2, 1)
    fig.add_trace(go.Scatter(x=h.time, y=h.tc_est, line=dict(color=EST, dash="dash"), name="Core temp (est.)"), 2, 1)
    fig.add_trace(go.Scatter(x=h.time, y=h.skin_obs, line=dict(color=QUIET, width=1), name="Skin temp"), 2, 1)
    fig.add_trace(go.Scatter(x=h.time, y=h.t_out, line=dict(color="#e67e22"), name="Outdoor °C"), 3, 1)
    fig.add_trace(go.Bar(x=h.time, y=np.where(h.outdoor, h.t_out.max(), np.nan), marker_color="rgba(230,126,34,.2)",
                         name="Outdoors", width=5 * 60 * 1000), 3, 1)
    fig.update_layout(height=620, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.05))
    st.plotly_chart(fig, width="stretch")
    full = tw.history()
    st.caption("How the twin learned this patient (narrowing band = growing confidence)")
    fg = go.Figure()
    for k, c in (("hr_gain", RISK),):
        fg.add_trace(go.Scatter(x=pd.concat([full.time, full.time[::-1]]),
                                y=pd.concat([full[k] + full[k + "_sd"], (full[k] - full[k + "_sd"])[::-1]]),
                                fill="toself", fillcolor="rgba(209,73,91,.15)", line=dict(width=0), showlegend=False))
        fg.add_trace(go.Scatter(x=full.time, y=full[k], line=dict(color=c, dash="dash"), name="HR response (bpm/°C)"))
    fg.update_layout(height=230, margin=dict(t=10, b=10), yaxis_title="bpm per °C")
    st.plotly_chart(fg, width="stretch")

# ------------------------------------------------------------------------------------ 3 forecast
def psi_figure(results: list[tuple[str, dict, str, str]], truth: pd.DataFrame | None = None):
    fig = go.Figure()
    for name, r, color, dash in results:
        df = r["df"]
        rgba = color.lstrip("#")
        rgba = f"rgba({int(rgba[:2], 16)},{int(rgba[2:4], 16)},{int(rgba[4:], 16)},.14)"
        fig.add_trace(go.Scatter(x=pd.concat([df.time, df.time[::-1]]), y=pd.concat([df.psi_hi, df.psi_lo[::-1]]),
                                 fill="toself", fillcolor=rgba, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.time, y=df.psi_mid, line=dict(color=color, dash=dash, width=2.2), name=name))
    if truth is not None:
        fig.add_trace(go.Scatter(x=results[0][1]["df"].time[:len(truth)], y=truth.psi, line=dict(color=OBS, width=2),
                                 name="Revealed truth (hidden from twin)"))
    fig.add_hline(y=7, line=dict(color=RISK, dash="dot"), annotation_text="High strain (PSI 7)")
    fig.update_layout(height=380, yaxis=dict(title="Physiological Strain Index (0–10)", range=[0, 10]),
                      margin=dict(t=20, b=10), legend=dict(orientation="h", y=-0.15))
    return fig


with tabs[2]:
    df = base["df"]
    win = df[df.psi_hi >= 7]
    xgb = preds.query("pid == @pid and evening == @day")["xgb_with_twin"] if preds is not None else pd.Series(dtype=float)
    c1, c2, c3 = st.columns(3)
    c1.metric("Twin: P(high strain in 24 h)", f"{base['p_event']:.0%}", band(base["p_event"]))
    if len(xgb):
        c2.metric("Risk model (XGBoost + twin)", f"{float(xgb.iloc[0]):.0%}", band(float(xgb.iloc[0])))
    c3.metric("Predicted peak PSI", f"{base['peak_psi']:.1f}")
    if len(win):
        st.error(f"High-risk window: {win.time.min():%a %H:%M} – {win.time.max():%H:%M}", icon="🔥")
    reveal = st.toggle("Reveal hidden truth (hide-and-reveal check)", value=False)
    st.plotly_chart(psi_figure([("Twin forecast", base, RISK, "dash")], tw.truth() if reveal else None),
                    width="stretch")
    st.subheader("Why: main drivers of tomorrow's risk")
    st.caption("Each factor is reset to a neutral value; the drop in predicted strain shows how much it contributes.")
    ex = pd.DataFrame(twin_at(pid, t_now).explain(base))
    ex["Peak PSI reduction if removed"] = ex["peak_psi_drop"].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}")
    ex["Risk reduction if removed"] = ex["risk_drop"].map(lambda v: "" if pd.isna(v) else f"{v:+.0%}")
    st.dataframe(ex[["driver", "Peak PSI reduction if removed", "Risk reduction if removed"]]
                 .rename(columns={"driver": "Driver"}), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------ 4 what-if
with tabs[3]:
    mode = st.radio("Mode", ["One patient, two scenarios", "Two patients, same heatwave"], horizontal=True)
    if mode.startswith("One"):
        l, r = st.columns([1, 2.4])
        with l:
            st.markdown("**Scenario**")
            preset = st.selectbox("Start from", list(PRESETS), index=1)
            p0 = PRESETS[preset]
            off = st.slider("Temperature vs forecast (°C)", -4.0, 6.0, float(p0.temp_offset_c), 0.5)
            ind = st.checkbox("Stay indoors and rest", value=p0.indoors_from_h is not None)
            hrs = st.slider("…between", 6, 22, (int(p0.indoors_from_h or 11), int(p0.indoors_to_h or 16)),
                            disabled=not ind)
            act = st.slider("Activity level", 0.4, 1.2, float(p0.activity_scale), 0.1)
            fl = st.checkbox("Extra fluids (drink every 90 min)", value=p0.extra_fluids)
            pc = st.checkbox("Power cut / no AC", value=p0.power_cut, disabled=not row["home_ac"],
                             help="Only meaningful for patients who have AC")
            st.markdown("**Hypothetical medication scenario**")
            st.caption("For clinical judgement only. Not a recommendation to change any medication.")
            nd = st.checkbox("If not on a diuretic", value=p0.no_diuretic, disabled=not row["diuretic"])
            ns = st.checkbox("If not on an SGLT2 inhibitor", value=p0.no_sglt2i, disabled=not row["sglt2i"])
            sc = Scenario("Scenario", off, pc, hrs[0] if ind else None, hrs[1] if ind else None, act, fl, nd, ns)
        with r:
            res = fc(pid, t_now, sc)
            dash = "dot" if sc.hypothetical else "dash"
            a, b, c = st.columns(3)
            a.metric("Peak PSI", f"{res['peak_psi']:.1f}", f"{res['peak_psi'] - base['peak_psi']:+.1f}",
                     delta_color="inverse")
            b.metric("P(high strain)", f"{res['p_event']:.0%}", f"{(res['p_event'] - base['p_event']) * 100:+.0f} pts",
                     delta_color="inverse")
            c.metric("Peak core temp", f"{res['peak_tc']:.2f} °C", f"{res['peak_tc'] - base['peak_tc']:+.2f}",
                     delta_color="inverse")
            if sc.hypothetical:
                st.info("Dotted line = hypothetical medication scenario. It shows how much of the modelled risk "
                        "is attributable to the drug's fluid effect; the decision stays with the clinician.")
            st.plotly_chart(psi_figure([("Baseline", base, QUIET, "dash"),
                                        ("Scenario", res, HYP if sc.hypothetical else EST, dash)]),
                            width="stretch")
    else:
        others = [p for p in pids if p != pid]
        pid2 = st.selectbox("Compare with", others,
                            format_func=lambda x: f"{x} · {cohort.set_index('pid').loc[x, 'name']}")
        r2 = cohort.set_index("pid").loc[pid2]
        same = st.checkbox(f"Expose both to {row['name']}'s forecast weather", value=True,
                           help="Same heatwave, different bodies: isolates what the EHR and learned traits change")
        b2 = same_weather_forecast(pid2, pid, t_now) if same else fc(pid2, t_now, PRESETS["Baseline"])
        st.plotly_chart(psi_figure([(f"{row['name']} ({pid})", base, RISK, "dash"),
                                    (f"{r2['name']} ({pid2})", b2, EST, "dash")]), width="stretch")
        comp = pd.DataFrame({
            "": ["Age", "eGFR", "Beta-blocker", "Diuretic", "SGLT2i", "Home AC", "Outdoor routine",
                 "Learned HR response", "Tomorrow max °C", "Peak PSI", "P(high strain)"],
            pid: [row.age, f"{row.egfr:.0f}", *("yes" if row[c] else "no" for c in
                  ("beta_blocker", "diuretic", "sglt2i", "home_ac", "outdoor_routine")),
                  f"{twin_at(pid, t_now).state()['hr_gain']:.0f}", f"{base['df'].t_out.max():.1f}",
                  f"{base['peak_psi']:.1f}", f"{base['p_event']:.0%}"],
            pid2: [r2.age, f"{r2.egfr:.0f}", *("yes" if r2[c] else "no" for c in
                   ("beta_blocker", "diuretic", "sglt2i", "home_ac", "outdoor_routine")),
                   f"{twin_at(pid2, t_now).state()['hr_gain']:.0f}", f"{b2['df'].t_out.max():.1f}",
                   f"{b2['peak_psi']:.1f}", f"{b2['p_event']:.0%}"]})
        st.dataframe(comp, hide_index=True, width="stretch")
        st.caption("Same weather for both patients." if same else "Each patient under their own city's forecast.")

# ------------------------------------------------------------------------------------ 5 evidence
with tabs[4]:
    st.subheader("What each number is")
    st.dataframe(pd.DataFrame({
        "Label": ["Observed", "Estimated", "Predicted", "Hypothetical"],
        "Meaning": ["Measured by the (simulated) wearable or taken from weather data",
                    "Inferred by the twin; never measured (core temperature, fluid deficit, personal traits)",
                    "Forecast for the next 24 h with a 90% uncertainty band",
                    "What-if output under changed assumptions, including medication scenarios"],
        "Examples": ["Heart rate, skin temp, steps, outdoor temp", "Core temp, HR response, fluid deficit",
                     "PSI curve, P(high strain)", "Stay indoors, no diuretic"]}), hide_index=True,
        width="stretch")
    st.subheader("Model performance (synthetic test patients)")
    for f, title in (("e1_core_temperature.csv", "E1 · Core-temperature estimation"),
                     ("e3_hide_and_reveal.csv", "E3 · Hide-and-reveal forecast"),
                     ("e4_risk_models.csv", "E4 · Risk models")):
        if (RESULTS / f).exists():
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(RESULTS / f).round(3), hide_index=True, width="stretch")
    st.subheader("Limits")
    st.markdown("""
- All data is synthetic. Results show the method works in our simulated world, not clinical accuracy.
- Physiology is simplified; drug effect sizes are illustrative assumptions (all in `config/assumptions.yaml`).
- PSI was designed for exertional heat strain in healthy people; used here as a research target.
- Sweat capacity and fluid-loss rate cannot be learned from wrist data alone; they stay at EHR-based estimates
  and their uncertainty is carried into every forecast.
- Medication scenarios show attributable risk only; they are never a recommendation.
""")
    with st.expander("All modelling assumptions"):
        st.code(CONFIG_PATH.read_text(), language="yaml")
