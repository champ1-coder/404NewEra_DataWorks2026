# DataWorks Challenge 2026 – Reservoir History Matching and Forecast

**Team:** _404newera_

## 1. The problem in plain words

A reservoir simulator turns four uncertain "dials" into oil, gas and water production curves, but each run is slow.
We are given 100 simulated cases (4 dials and their 1998–2008 curves) plus the real field's production for 1998–2008.
The task:

1. Build a **fast proxy** of the simulator.
2. Find the dial settings that **match the real field's history**.
3. **Forecast 2008–2028** for the 10 best cases and fill the submission template.

The four dials are fault transmissibility, porosity multiplier, permeability multiplier and aquifer pore volume.

## 2. What is in this repository

| File | Purpose |
|---|---|
| `proxy_model.py` | Step 1 – trains the proxy (4 dials → 6 production curves) and tests it |
| `history_matching.py` | Step 2 – searches for dial settings that match the real history, and tests how well history pins each dial down |
| `forecast.py` | Step 3 – ranks the 100 cases, forecasts 2008–2028, fills the template, exports data for Power BI |

Input data is **not** included. Put these three files in a `data/` folder (exact names):
`01_Introduction.xlsx`, `02_Field_Production.xlsx`, `09_Template_Deliverable.xlsx`.
The large rock-property files (`03`–`08`) were inspected but are not used: the proxy uses the 4 dials per case.

```
dataworks/
  proxy_model.py  history_matching.py  forecast.py  README.md
  data/
    01_Introduction.xlsx
    02_Field_Production.xlsx
    09_Template_Deliverable.xlsx
```

## 3. How to run (Windows, from inside the `dataworks` folder)

```
pip install pandas numpy scikit-learn scipy matplotlib joblib openpyxl
python proxy_model.py data proxy_results
python history_matching.py data proxy_results matching_results
python forecast.py data data/09_Template_Deliverable.xlsx forecast_results
```

Run them **in this order** (step 2 loads the model saved by step 1). Random seeds are fixed, so results are reproducible.

| Output folder | Main files |
|---|---|
| `proxy_results/` | `proxy_model.joblib`, validation and test scores, accuracy plots |
| `matching_results/` | `matched_dials.csv`, `matched_curves.csv`, `dial_identifiability.csv`, `history_match.png` |
| `forecast_results/` | **`09_Template_Deliverable_filled.xlsx`** (the submission), `forecast.png`, `backtest_grid.csv`, `best10_summary.csv`, `forecast_all_cases_for_powerbi.csv` |

## 4. Method

**Step 1 – proxy.** Cases 1–70 train, 71–85 validate, 86–100 test (scored once, at the end). Three models were compared:
linear, quadratic, and PCA + Gaussian process. All six curves (oil, gas, water; cumulative and rate) are predicted together.
The best on validation (PCA + Gaussian process) is refitted on cases 1–85 and saved.

**Step 2 – history matching.** Mismatch = scaled RMS error between predicted and observed curves (each curve type scaled so no fluid dominates).
We try 200,000 dial settings through the proxy, refine with an optimiser, then keep every setting that fits about as well as the best.
A recovery test checks whether the dials of unseen cases can be found from their curves.

**Step 3 – forecast.** The 10 best-matching existing cases (recent years weighted more) have their rates extended to 2028,
then converted to cumulative production starting from the **real field's** 2008 values. The extension model is chosen by a
**backtest** (hide the last 2–4 years, forecast them, compare), and models whose water cut hits an artificial 98% ceiling
within 20 years are excluded. Chosen model: power-law decline of total liquid with a linearly growing water-oil ratio, fitted on the last 6 quarters.

## 5. Results

| | Result |
|---|---|
| Proxy accuracy (test cases 86–100) | R² ≥ 0.998 on every curve; final-date cumulative error 0.04% (oil, gas), 0.24% (water) |
| Proxy speed | 10,000 dial settings predicted in about 0.1 s |
| Best history match | mismatch 0.135 (best of the 100 given cases: 0.138) |
| Best 10 cases | Case 6, 73, 76, 97, 11, 40, 50, 20, 90, 26 |
| Forecast backtest error (chosen model) | about 5% on the 100 cases, about 7% on the real field |
| Cumulative by 2028 (primary model) | oil 60.7–61.3 M STB, gas 22.1–22.3 M MSCF, water 87.6–88.2 M STB |

## 6. Key findings and data notes

- **Observed rates are shifted one quarter.** The observed oil/gas rate on a date equals the simulated rate one quarter *later*;
  cumulative curves line up on the same date. All code compares observed rate at `t` with simulated rate at `t+1`.
  Ignoring this makes the mismatch look about seven times worse and selects the wrong cases.
- **History pins down only porosity.** Porosity multiplier is well determined; permeability and aquifer volume only roughly;
  fault transmissibility essentially not at all. Many dial settings fit the past equally well, so we report uncertainty rather than one "true" answer.
- **The real field's last two years decline faster than any allowed dial setting can reproduce** (last oil rate about 5,250 STB/d vs about 5,800 at best),
  so the forecast starts from the observed values rather than the simulated case's.
- **Model choice matters more than case choice.** The 10 cases agree within about 1% by 2028; alternative extension models give 2028 oil
  from about 52.6 M to 61 M STB. The scenarios are exported for the dashboard.

## 7. Requirements

Python 3.9+, pandas, numpy, scikit-learn, scipy, matplotlib, joblib, openpyxl.
