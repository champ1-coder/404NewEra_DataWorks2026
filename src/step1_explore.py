"""
DataWorks 2026 - Step 1: explore the simulated cases vs the observed field history.
Put this file in the same folder as 02_Field_Production.xlsx and run:  python step1_explore.py
Needs: pip install pandas numpy matplotlib openpyxl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = "02 Field Production.xlsx"

# Sheets with one column per case (Case_1 ... Case_100) + a column called RAMP_History (the observed field data)
SHEETS = {
    "oil_rate":   "Oil production rate",
    "gas_rate":   "Gas production rate",
    "water_rate": "Water production rate",
}

def load_sheet(sheet):
    # row 0 = X/Y labels, row 1 = column names, row 2 = units, data starts at row 3
    df = pd.read_excel(FILE, sheet_name=sheet, header=1, skiprows=[2])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").apply(pd.to_numeric, errors="coerce")

def to_history_dates(df, dates):
    """Each case only has values on its own simulator time steps (the rest is NaN).
    Interpolate every column onto a common set of dates so cases can be compared."""
    out = {}
    x_new = dates.values.astype("datetime64[s]").astype(float)
    for c in df.columns:
        s = df[c].dropna()
        x = s.index.values.astype("datetime64[s]").astype(float)
        out[c] = np.interp(x_new, x, s.values)
    return pd.DataFrame(out, index=dates)

# 1) observed history (quarterly, 1998-2008)
hist = pd.read_excel(FILE, sheet_name="Field Production")
hist["Date"] = pd.to_datetime(hist["Date"])
dates = pd.DatetimeIndex(hist["Date"])

# 2) simulated cases on the same quarterly dates
data = {k: to_history_dates(load_sheet(s), dates) for k, s in SHEETS.items()}

# 3) mismatch of each case vs the observed history (RAMP_History column)
scores = {}
for k, df in data.items():
    df = df[df.index >= "1999-01-01"]   # skip the 1998 start-up step: interpolating across it adds the same fake error to every case
    obs = df["RAMP_History"]
    cases = df.drop(columns="RAMP_History")
    scores[k] = np.sqrt(((cases.sub(obs, axis=0)) ** 2).mean()) / obs.max()   # normalised RMSE
scores = pd.DataFrame(scores)
scores["total"] = scores.mean(axis=1)
scores = scores.sort_values("total")
print("10 cases that already match history best (lower = better):")
print(scores.head(10).round(4))
scores.to_csv("case_mismatch_scores.csv")

# 4) plot every case vs history
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (k, df) in zip(axes, data.items()):
    for c in df.columns.drop("RAMP_History"):
        ax.plot(df.index, df[c], color="lightgray", lw=0.6)
    for c in scores.index[:10]:
        ax.plot(df.index, df[c], lw=0.9)
    ax.plot(df.index, df["RAMP_History"], "k--", lw=2, label="Observed history")
    ax.set_title(k); ax.legend()
plt.tight_layout(); plt.savefig("cases_vs_history.png", dpi=120)
print("saved case_mismatch_scores.csv and cases_vs_history.png")
