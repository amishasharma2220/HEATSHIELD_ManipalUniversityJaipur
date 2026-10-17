# HEATSHIELD — a medication-aware digital twin for heat strain in vulnerable patients

> **Happiest Health Digital Twin Challenge 2026** · Prototype on synthetic data · **Not validated for clinical use**

![Core-temperature error: standard method vs HEATSHIELD](results/e1_core_temperature.png)

## Team

| | |
| --- | --- |
| Team name | HEATSHIELD |
| Members | Amisha Sharma (team leader) |
| College / incubator | Manipal University Jaipur |
| Contact | amishasharma2220@gmail.com |

## Project title

**HEATSHIELD: a patient-specific, medication-aware digital twin that forecasts heat strain 24 hours ahead and lets a clinician test what-if scenarios.**

## Problem statement and healthcare use case

Indian heatwaves routinely push daytime temperatures into the mid-40s °C. Older patients with hypertension, type 2 diabetes or chronic kidney disease are the most vulnerable, and several of their everyday medicines make heat more dangerous:

| Drug class | Effect in heat |
| --- | --- |
| Diuretics (hydrochlorothiazide, chlorthalidone) | Extra fluid loss: the patient dehydrates faster |
| SGLT2 inhibitors (dapagliflozin, empagliflozin) | Glucose in urine pulls water with it |
| Beta-blockers (metoprolol, atenolol) | Blunt the heart-rate rise, hiding the usual warning sign |

Wearables cannot measure core body temperature, so heat-strain monitors estimate it from heart rate. **That assumption breaks for beta-blocked patients**: core temperature rises but heart rate doesn't follow, so the standard method underestimates strain in exactly the people most at risk.

**Use case.** A clinician or care team monitoring vulnerable patients during a heatwave. Each evening HEATSHIELD ranks patients by tomorrow's risk. For any patient, the doctor sees the twin's estimate of their current state, a 24-hour strain forecast with uncertainty, the main drivers, and can test scenarios: *stay indoors 11am–4pm*, *hotter day*, *power cut*, *lighter activity*, *extra fluids*, or a clearly labelled *hypothetical* medication scenario.

**Adverse event predicted.** High heat strain in the next 24 hours, defined as Physiological Strain Index (PSI, Moran et al. 1998) ≥ 7 **or** core temperature ≥ 39.0 °C. The temperature arm stops a drug-blunted heart rate from hiding real strain.

### What makes it a digital twin, not a classifier

| | Typical ML model | HEATSHIELD |
| --- | --- | --- |
| Learns | Population patterns | This patient's own heat response |
| Output | A risk score now | A forecast trajectory with an uncertainty band |
| Updates | Retrained occasionally | Every 5 minutes, with each wearable reading |
| Answers "what if?" | No | Yes: re-simulates the patient under changed inputs |
| Needs labelled outcomes | Yes | No: the forecast is mechanistic |

## Architecture

![Architecture](docs/architecture.png)

1. **World simulator** (`src/heatshield/world/`): the "fake reality". A rich heat + fluid balance generates each patient's wearable stream and hidden ground truth. It has per-patient hidden traits (sweat capacity, heart-rate response to heat, fluid-loss rate, fluid reserve, resting HR and core temperature), a circadian rhythm, humidity-limited sweating, solar load, sensor noise and dropouts. **The twin never sees any of this**: it is used only to grade the twin.
2. **The twin** (`src/heatshield/twin/`): one virtual patient per person.
   - *EHR priors:* age, eGFR, weight and the medication list set starting beliefs (e.g. beta-blocker → low heart-rate response; diuretic/SGLT2i → extra fluid loss; low eGFR → smaller fluid reserve).
   - *Particle filter:* 400 particles per patient track core temperature and fluid deficit from heart rate, skin temperature, activity, phone location (indoor/outdoor) and weather, and learn the patient's heart-rate response and resting HR online (Liu–West parameter learning). The twin's own physics is deliberately **simpler** than the world's, so it cannot read its own answers back.
   - *Forecast:* every particle is run 24 h forward under the weather forecast and the patient's usual routine, giving a PSI curve, a 90% band and P(high strain).
   - *Scenarios and explanations:* inputs are changed and the forecast re-run; drivers are ranked by how much risk drops when each is reset.
3. **Risk model** (`src/heatshield/risk/`): XGBoost on EHR + wearable + weather (+ twin) features, patient-level split, Platt calibration. Used as a benchmark and second opinion.
4. **Clinician dashboard** (`app/dashboard.py`): Streamlit + Plotly. Triage list; per-patient tabs for overview, live twin state, 24 h forecast with drivers, what-if simulator (one patient × two scenarios, or two patients × the same heatwave), and evidence & limits.

Production path (not built): smartwatch APIs for the wearable stream, a FHIR feed for the EHR, a live weather-forecast API, a time-series database (PostgreSQL/TimescaleDB) for twin states.

## Data (all synthetic or open; no real patient data)

| Stream | Source in this repo | Notes |
| --- | --- | --- |
| EHR | `heatshield.ehr` writes **Synthea-format** CSVs (`patients`, `conditions`, `medications`, `observations`) for 240 adults aged 55–85 | Real Synthea output can be dropped into `data/raw/ehr/` instead; the same loader maps it. India-specific fields added: home AC, outdoor routine |
| Weather | `heatshield.weather` — Open-Meteo historical archive, May–June 2024 heatwave, 5 cities (Jaipur, Delhi, Jodhpur, Nagpur, Lucknow) | `python -m heatshield.weather --fetch` downloads real data. If offline, a labelled synthetic heatwave is used; `data/raw/weather/SOURCE.txt` records which |
| Wearable | Generated by the world simulator every 5 min: heart rate, skin temperature, steps, sleep, indoor/outdoor | Sensor noise and 2% dropouts |

Every physiological and drug assumption is in **one file**: [`config/assumptions.yaml`](config/assumptions.yaml). All effect sizes are illustrative, not clinical reference values.

## Technical stack and AI/ML details

| Layer | Tools |
| --- | --- |
| Language | Python 3.11+ |
| State estimation | NumPy particle filter (vectorised over patients × particles), Liu–West parameter learning; Buller et al. (2013) extended Kalman filter as baseline |
| Physiology | Heat balance (metabolic, dry exchange, humidity-limited sweating, regulation) + fluid balance |
| Risk model | XGBoost, scikit-learn (Platt calibration, metrics) |
| Dashboard | Streamlit, Plotly |
| Tests | pytest (scenario sanity checks) |

## Results (synthetic test patients, 240-patient cohort)

**E1 · Medication-awareness matters.** Core-temperature error during heat strain (true core ≥ 37.8 °C):

| Method | Beta-blocked | Not beta-blocked |
| --- | --- | --- |
| Standard HR method (Buller 2013) | 1.05 °C | 0.43 °C |
| Population model (no EHR) | 0.33 °C | 0.24 °C |
| HEATSHIELD twin | **0.21 °C** | **0.21 °C** |

The standard method underestimates beta-blocked patients' core temperature by 1.0 °C on average during heat strain. See `results/e1_example_trace.png`.

**E2 · The twin learns the patient.** Error in the patient's heart-rate response to heat falls from 10.5 bpm/°C (population average) to 3.0 (EHR prior) to 2.4 after the wearable data (correlation with the hidden truth 0.97); resting-HR error falls from 2.5 to 0.6 bpm.
*Honest finding:* sweat capacity and fluid-loss rate are **not identifiable** from wrist heart rate and skin temperature; learning them made estimates worse. They stay at EHR-based estimates and their uncertainty is carried into every forecast.

**E3 · Hide-and-reveal.** Calibrate on days 1–5, forecast days 6–7 with no sensor data, then reveal the truth:

| Model | Core-temp RMSE | Peak-PSI error | 90% band coverage |
| --- | --- | --- | --- |
| Population model | 0.16 °C | 0.99 | 77% |
| EHR-aware, no learning | 0.15 °C | 0.44 | 83% |
| **HEATSHIELD twin** | **0.15 °C** | **0.33** | **83%** |

![Hide and reveal](results/e3_hide_and_reveal.png)

**E4 · Predicting tomorrow's high strain** (1,680 patient-evenings, 12.7% events; mean over 5 random patient splits):

| Model | AUROC | AUPRC | Brier |
| --- | --- | --- | --- |
| Heat-index rule | 0.63 | 0.20 | 0.67 |
| XGBoost, no twin | 0.92 | 0.60 | 0.07 |
| **Twin forecast alone** | **0.94** | 0.60 | 0.09 |
| XGBoost + twin features | 0.94 | **0.66** | **0.07** |

The twin's mechanistic forecast matches a supervised model trained on 1,000+ labelled patient-days, **without needing any labelled outcomes**, which a real clinic would not have. That is why the dashboard's primary risk is the twin's own forecast.

**E5 · The scenario engine behaves sensibly** (`tests/test_scenarios.py`, 9 checks, all pass): a hotter day never lowers strain; staying indoors, lighter activity and extra fluids never raise it; a power cut only matters for patients with AC; the diuretic scenario only changes diuretic patients; PSI formula endpoints.

## How to run

```bash
pip install -r requirements.txt
python -m heatshield.weather --fetch      # optional: real Open-Meteo weather (needs internet; run from src/ or set PYTHONPATH=src)
python experiments/run_all.py              # data -> world -> twin -> E1-E4 -> results/  (~4 min)
pytest -q                                  # E5 scenario checks
streamlit run app/dashboard.py             # clinician dashboard
```

`python experiments/run_all.py --quick` runs a 60-patient smoke test in under a minute.

## Repository layout

```
config/assumptions.yaml     every modelling assumption in one place
src/heatshield/
  ehr.py                    Synthea-format EHR generator + loader
  weather.py                Open-Meteo download, synthetic fallback, indoor model, heat index
  schedule.py               daily routines (activity, location, drinking)
  psi.py                    Physiological Strain Index
  world/simulate.py         world simulator (ground truth, hidden from the twin)
  twin/model.py             the twin's simplified physics + EHR priors
  twin/filter.py            particle filter, Buller baseline, forecasting
  twin/scenarios.py         what-if scenarios
  twin/twin.py              PatientTwin class used by the dashboard
  risk/features.py, train.py
experiments/run_all.py      reproduces every result and figure
tests/test_scenarios.py     E5
app/dashboard.py            Streamlit app
results/                    metrics (CSV) and figures (PNG)
docs/                       architecture diagram and presentation
```

## Demo video

_TODO: unlisted YouTube link (≥ 20 minutes)_

## Documents

- Architecture diagram: [`docs/architecture.pdf`](docs/architecture.pdf) _(TODO)_
- Presentation: [`docs/presentation.pdf`](docs/presentation.pdf) _(TODO)_

## Limitations and responsible use

- All data is synthetic. The results show the method works in a simulated world under stated assumptions, not clinical accuracy.
- The physiology is simplified; drug effect sizes are illustrative.
- PSI was designed for exertional heat strain in healthy people; it is used here as a research target, not a diagnosis.
- Medication scenarios show **attributable risk only**. HEATSHIELD never recommends stopping or changing a medicine.
- Real deployment needs validation on real wearable and EHR data, with ethics approval and DPDP-compliant data handling.

## Related work

Personal heat-strain digital twins exist for **healthy** workers, soldiers and athletes (e.g. NTU Singapore's construction-worker twin, the ThermRisk project). A systematic review of 25 wearable core-temperature algorithms found few used individual health data. HEATSHIELD's contribution is making the twin **patient-specific through the EHR**, and in particular correcting for beta-blockers, which break the standard heart-rate method.

**Key references** (verify citations before submission): Moran, Shitzer & Pandolf (1998) *Am J Physiol* 275:R129 (PSI); Buller et al. (2013) *Physiol Meas* 34:781 (core temperature from heart rate); Liu & West (2001) in *Sequential Monte Carlo Methods in Practice* (parameter learning); Dolson et al. (2022) *Sensors* 22:7639 (systematic review).

## License

[MIT](LICENSE)
