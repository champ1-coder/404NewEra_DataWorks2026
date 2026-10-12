"""
DataWorks 2026 - Step 3: pick the best 10 cases, extend them to 2028, fill the template.
Run from the folder with the Excel files:  python step3_forecast.py
Needs: pip install pandas numpy matplotlib openpyxl
"""
import numpy as np, pandas as pd, openpyxl
import matplotlib.pyplot as plt

FIELD    = "02 Field Production.xlsx"          # <- use underscores here if your files have them
TEMPLATE = "09 Template Deliverable.xlsx"
OUT      = "09 Template Deliverable - filled.xlsx"
RATE = {"oil": "Oil production rate", "gas": "Gas production rate", "water": "Water production rate"}
CUM  = {"oil": "Oil production cumulative", "gas": "Gas production cumulative", "water": "Water production cumulative"}
N_FIT = 8            # fit the trend on the last 8 quarters (2 years) of history
N_BEST = 10
import os
WATER_MODEL = os.environ.get("WATER_MODEL", "wor")   # "wor" = smooth curve that levels off (default) | "linear" = old straight-line water cut

# ---------------- 1. load curves on the quarterly history dates ----------------
hist  = pd.read_excel(FIELD, sheet_name="Field Production")
dates = pd.DatetimeIndex(pd.to_datetime(hist["Date"]))
xn = dates.values.astype("datetime64[s]").astype(float)

def load(sheet):
    df = pd.read_excel(FIELD, sheet_name=sheet, header=1, skiprows=[2])
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").apply(pd.to_numeric, errors="coerce")
    out = {c: np.interp(xn, df[c].dropna().index.values.astype("datetime64[s]").astype(float), df[c].dropna().values) for c in df.columns}
    return pd.DataFrame(out, index=dates)

rates = {k: load(s) for k, s in RATE.items()}
cums  = {k: load(s) for k, s in CUM.items()}
cases = [c for c in rates["oil"].columns if c.startswith("Case_")]

# ---------------- 2. score every case against the observed history ----------------
# recent years count more, because the forecast starts where the history ends
keep = dates >= "1999-01-01"
w = np.linspace(1, 4, keep.sum())
score = pd.Series(0.0, index=cases)
for k in RATE:
    obs = rates[k]["RAMP_History"][keep].values; sc = obs.max()
    for c in cases:
        score[c] += np.sqrt(np.average(((rates[k][c][keep].values - obs) / sc) ** 2, weights=w)) / 3
score = score.sort_values()
best = list(score.index[:N_BEST])
print("Best 10 cases (lower score = closer to the real history):"); print(score.head(N_BEST).round(4))

# ---------------- 3. extend rates forward (simple, explainable trends) ----------------
def extend(t_hist, qo, qg, qw, t_fut):
    """t in years since the last history date.
       total liquid (oil+water): exponential decline
       water cut: the water-oil ratio (WOR) is extended as an exponential trend, i.e. ln(WOR) is a straight line in time.
                  Water cut = WOR/(1+WOR), so it rises quickly then levels off towards 100% by itself (no artificial cap).
       oil = liquid x (1 - water cut), water = liquid x water cut | gas = oil x recent gas-oil ratio"""
    tw, qo_w, qg_w, qw_w = t_hist[-N_FIT:], qo[-N_FIT:], qg[-N_FIT:], qw[-N_FIT:]
    ql_w = qo_w + qw_w
    s, b = np.polyfit(tw, np.log(ql_w), 1);  ql_f = np.exp(b + s * t_fut)
    if WATER_MODEL == "linear":                              # old method, kept only for comparison
        s2, b2 = np.polyfit(tw, qw_w / ql_w, 1);  fw_f = np.clip(b2 + s2 * t_fut, 0, 0.98)
    else:
        s2, b2 = np.polyfit(tw, np.log(qw_w / qo_w), 1);  wor_f = np.exp(b2 + s2 * t_fut);  fw_f = wor_f / (1 + wor_f)
    qo_f, qw_f = ql_f * (1 - fw_f), ql_f * fw_f
    qg_f = np.mean(qg_w / qo_w) * qo_f
    return qo_f, qg_f, qw_f

def to_cum(c0, q_last, q_f, t_last, t_fut):
    days = np.diff(np.r_[t_last, t_fut]) * 365.25
    q = np.r_[q_last, q_f]
    return c0 + np.cumsum(0.5 * (q[:-1] + q[1:]) * days)

future = pd.date_range("2008-04-01", "2028-01-01", freq="3MS")
t_fut  = (future - dates[-1]).days.values / 365.25
t_h    = (dates - dates[-1]).days.values / 365.25

def forecast_case(c):
    qo, qg, qw = rates["oil"][c].values, rates["gas"][c].values, rates["water"][c].values
    fo, fg, fw = extend(t_h, qo, qg, qw, t_fut)
    return pd.DataFrame({
        "Gas production cumulative [MSCF]":  to_cum(cums["gas"][c].iloc[-1],   qg[-1], fg, 0.0, t_fut),
        "Oil production cumulative [STB]":   to_cum(cums["oil"][c].iloc[-1],   qo[-1], fo, 0.0, t_fut),
        "Water production cumulative [STB]": to_cum(cums["water"][c].iloc[-1], qw[-1], fw, 0.0, t_fut),
        "Oil rate [STB/d]": fo, "Gas rate [MSCF/d]": fg, "Water rate [STB/d]": fw}, index=future)

# ---------------- 4. honest check: hide the last 3 years, forecast them, compare ----------------
cut = int(np.where(dates == "2005-01-01")[0][0])
err = {"oil": [], "gas": [], "water": []}
for c in cases:
    th = t_h[: cut + 1] - t_h[cut]; tf = t_h[cut + 1:] - t_h[cut]
    q = {k: rates[k][c].values for k in RATE}
    fo, fg, fw = extend(th, q["oil"][: cut + 1], q["gas"][: cut + 1], q["water"][: cut + 1], tf)
    for k, f in zip(("oil", "gas", "water"), (fo, fg, fw)):
        c0 = cums[k][c].iloc[cut]; pred = to_cum(c0, q[k][cut], f, 0.0, tf)[-1] - c0
        err[k].append(pred / (cums[k][c].iloc[-1] - c0) - 1)
print("\nBacktest: forecast Jan 2005 -> Jan 2008 and compare with the simulated truth (error of the 3-year production):")
for k in err: print(f"  {k:6s} mean {np.mean(err[k]) * 100:+.1f}%   typical |error| {np.mean(np.abs(err[k])) * 100:.1f}%   worst {np.max(np.abs(err[k])) * 100:.1f}%")

# ---------------- 5. fill the template (one sheet per case) ----------------
wb = openpyxl.load_workbook(TEMPLATE)
rows, mapping = [], []
for i, c in enumerate(best, start=1):
    f = forecast_case(c); ws = wb[f"Case {i}"]
    header = [cell.value for cell in ws[1]]
    n_rows = sum(1 for r in ws.iter_rows(min_row=2, max_col=1, values_only=True) if r[0] is not None)
    assert len(f) == n_rows, f"template Case {i} has {n_rows} date rows, expected {len(f)}"
    for r, (_, row) in enumerate(f.iterrows(), start=2):
        for col, name in enumerate(header, start=1):
            if name in f.columns: ws.cell(row=r, column=col, value=float(row[name]))
    mapping.append({"Template sheet": f"Case {i}", "Simulation case": c, "Mismatch score": score[c]})
    h = pd.DataFrame({"Oil rate [STB/d]": rates["oil"][c], "Gas rate [MSCF/d]": rates["gas"][c], "Water rate [STB/d]": rates["water"][c],
                      "Oil production cumulative [STB]": cums["oil"][c], "Gas production cumulative [MSCF]": cums["gas"][c],
                      "Water production cumulative [STB]": cums["water"][c]}, index=dates); h["Period"] = "History (simulated)"
    f["Period"] = "Forecast"; both = pd.concat([h, f]); both.insert(0, "Rank", i); both.insert(0, "Case", c); rows.append(both)
wb.save(OUT)
long = pd.concat(rows); long.index.name = "Date"; long.reset_index().to_csv("forecast_all_cases_for_powerbi.csv", index=False)
pd.DataFrame(mapping).to_csv("best10_summary.csv", index=False)
print(f"\nSaved: {OUT}, forecast_all_cases_for_powerbi.csv, best10_summary.csv, forecast.png")

# ---------------- 6. picture ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a, (k, lab) in zip(ax, [("Oil production cumulative [STB]", "Oil cumulative"), ("Gas production cumulative [MSCF]", "Gas cumulative"), ("Water production cumulative [STB]", "Water cumulative")]):
    for c in best:
        d = long[long["Case"] == c]; a.plot(d.index, d[k], lw=1)
    a.axvline(dates[-1], color="k", ls=":"); a.set_title(lab)
plt.tight_layout(); plt.savefig("forecast.png", dpi=120)
