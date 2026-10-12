"""
DataWorks 2026 - Step 2: learn how the 4 parameters change the production curves (the "proxy"),
then search for the parameters that best match the observed history.
Run from the folder that has the Excel files:  python step2_proxy.py
Needs: pip install pandas numpy matplotlib openpyxl scikit-learn scipy
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import KFold
from scipy.optimize import differential_evolution

FIELD = "02 Field Production.xlsx"      # <- use "02_Field_Production.xlsx" if your file has underscores
INTRO = "01 Introduction.xlsx"          # <- same here
SHEETS = {"oil": "Oil production rate", "gas": "Gas production rate", "water": "Water production rate"}

# ---------- load inputs (4 parameters per case) ----------
X = pd.read_excel(INTRO, sheet_name="Uncertainty Values").set_index("Item")
X.index = [i.replace("Case ", "Case_") for i in X.index]          # "Case 1" -> "Case_1"
print("Inputs:", X.shape, list(X.columns))

# ---------- load outputs on the quarterly history dates ----------
hist = pd.read_excel(FIELD, sheet_name="Field Production")
dates = pd.DatetimeIndex(pd.to_datetime(hist["Date"]))
keep = dates >= "1999-01-01"                                       # skip the 1998 start-up step
xn = dates.values.astype("datetime64[s]").astype(float)

def load(sheet):
    df = pd.read_excel(FIELD, sheet_name=sheet, header=1, skiprows=[2])
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").apply(pd.to_numeric, errors="coerce")
    out = {}
    for c in df.columns:
        s = df[c].dropna(); out[c] = np.interp(xn, s.index.values.astype("datetime64[s]").astype(float), s.values)
    return pd.DataFrame(out, index=dates)[keep]

curves = {k: load(s) for k, s in SHEETS.items()}
scale = {k: curves[k]["RAMP_History"].max() for k in curves}       # so oil, gas, water count equally
def stack(col):  return np.concatenate([curves[k][col].values / scale[k] for k in SHEETS])
Y = np.array([stack(c) for c in X.index])                          # 100 cases x (3 curves x time)
y_obs = stack("RAMP_History")

# ---------- proxy: parameters -> PCA scores -> Gaussian Process ----------
lo, hi = X.min().values, X.max().values
Xs = (X.values - lo) / (hi - lo)                                   # scale every parameter to 0..1
def fit(Xtr, Ytr, n_comp=6):
    pca = PCA(n_components=n_comp).fit(Ytr); Z = pca.transform(Ytr)
    k = C(1.0) * RBF(length_scale=np.ones(Xtr.shape[1])) + WhiteKernel(1e-3)
    gps = [GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2).fit(Xtr, Z[:, j]) for j in range(n_comp)]
    return lambda Xn: pca.inverse_transform(np.column_stack([g.predict(Xn) for g in gps]))

# honest check: train on 80 cases, predict the 20 it never saw (5 times)
err = []
for tr, te in KFold(5, shuffle=True, random_state=0).split(Xs):
    pred = fit(Xs[tr], Y[tr])(Xs[te]); err.append(np.sqrt(((pred - Y[te]) ** 2).mean(axis=1)))
err = np.concatenate(err)
print(f"Proxy cross-validated error (normalised RMSE): mean {err.mean():.4f}, worst case {err.max():.4f}")
proxy = fit(Xs, Y)

# ---------- history matching: search the 4-parameter space for the best match ----------
def mismatch(p):  return np.sqrt(((proxy(p.reshape(1, -1))[0] - y_obs) ** 2).mean())
res = differential_evolution(mismatch, bounds=[(0, 1)] * 4, seed=0, tol=1e-8, maxiter=200)
best = lo + res.x * (hi - lo)
print("\nBest parameters found by the proxy:")
for n, v in zip(X.columns, best): print(f"  {n:26s} {v:.4f}")
print(f"Mismatch of best parameters: {res.fun:.4f}")
best_case = np.sqrt(((Y - y_obs) ** 2).mean(axis=1)); best_case = pd.Series(best_case, index=X.index).sort_values()
print("Best existing simulated case for comparison:", best_case.index[0], f"{best_case.iloc[0]:.4f}")
best_case.to_csv("case_mismatch_scores_step2.csv")

# ---------- plot ----------
n = len(curves["oil"]); t = curves["oil"].index
fig, ax = plt.subplots(1, 3, figsize=(16, 4)); pb = proxy(res.x.reshape(1, -1))[0]
for i, k in enumerate(SHEETS):
    sl = slice(i * n, (i + 1) * n)
    ax[i].plot(t, y_obs[sl] * scale[k], "k--", lw=2, label="Observed")
    ax[i].plot(t, pb[sl] * scale[k], "r", lw=2, label="Proxy at best parameters")
    ax[i].set_title(k + " rate"); ax[i].legend()
plt.tight_layout(); plt.savefig("history_match.png", dpi=120); print("saved history_match.png")
