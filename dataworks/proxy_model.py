"""
DataWorks Challenge 2026 - Step 1: Proxy model
4 reservoir "dials" (uncertainty values) -> 10 years of production curves (1998-2008).

Run:   python proxy_model.py [data_folder] [output_folder]
Needs: pandas, numpy, scikit-learn, matplotlib, joblib, openpyxl
Files expected in data_folder: 01_Introduction.xlsx, 02_Field_Production.xlsx
Cases 1-70 = train, 71-85 = validation, 86-100 = test (test is scored once, at the end).
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 0

# (short name, sheet name in 02_Field_Production.xlsx) - first 3 are cumulative, last 3 are rates
CURVES = [
    ("Oil cumulative", "Oil production cumulative"),
    ("Gas cumulative", "Gas production cumulative"),
    ("Water cumulative", "Water production cumulative"),
    ("Oil rate", "Oil production rate"),
    ("Gas rate", "Gas production rate"),
    ("Water rate", "Water production rate"),
]
NAMES = [c[0] for c in CURVES]
N_CURVES, N_T = len(CURVES), 41  # 41 quarterly history dates


def to_days(series):
    return pd.to_datetime(series).values.astype("datetime64[D]").astype(int).astype(float)


def load_data(data_dir):
    """Returns dates, X (100x4 dials), Y (100 x 246 curves), H (246 observed history), dial names."""
    xl = pd.ExcelFile(data_dir / "02_Field_Production.xlsx")
    dates = pd.to_datetime(pd.read_excel(xl, "Field Production")["Date"])
    t_hist = to_days(dates)
    cases = [f"Case_{i}" for i in range(1, 101)]
    y_blocks, h_blocks = [], []
    for _, sheet in CURVES:
        # sheet has a title row on top and a units row under the header -> header=1, skip row 2
        d = pd.read_excel(xl, sheet, header=1, skiprows=[2])
        t = to_days(d.iloc[:, 0])

        def on_history_dates(col):  # case dates are irregular -> interpolate onto the 41 history dates
            ok = d[col].notna().values
            return np.interp(t_hist, t[ok], d[col].values[ok].astype(float))

        y_blocks.append(np.column_stack([on_history_dates(c) for c in cases]))  # 41 x 100
        h_blocks.append(on_history_dates("RAMP_History"))
    Y = np.vstack(y_blocks).T  # 100 x (6*41)
    H = np.concatenate(h_blocks)
    u = pd.read_excel(data_dir / "01_Introduction.xlsx", sheet_name="Uncertainty Values").set_index("Item")
    u = u.loc[[f"Case {i}" for i in range(1, 101)]]
    return dates, u.values.astype(float), Y, H, list(u.columns)


class Proxy:
    """Small, fast model. kind = 'linear' | 'quadratic' | 'pca_gp'."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, Y):
        n = len(Y)
        self.xs = StandardScaler().fit(X)
        self.mu = Y.mean(0)
        # one scale per curve type, so no fluid dominates the fit
        sd = Y.reshape(n, N_CURVES, N_T).std(axis=0).max(axis=1) + 1e-9
        self.sd = np.repeat(sd, N_T)
        Xn, Yn = self.xs.transform(X), (Y - self.mu) / self.sd
        if self.kind == "pca_gp":
            self.pca = PCA(8, random_state=SEED).fit(Yn)
            Z = self.pca.transform(Yn)
            kernel = C(1.0) * RBF([1.0] * X.shape[1]) + WhiteKernel(1e-4)
            self.gps = [
                GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=SEED).fit(Xn, Z[:, i])
                for i in range(Z.shape[1])
            ]
        else:
            degree = 1 if self.kind == "linear" else 2
            self.reg = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3)).fit(Xn, Yn)
        return self

    def predict(self, X):
        Xn = self.xs.transform(np.atleast_2d(X))
        if self.kind == "pca_gp":
            Yn = self.pca.inverse_transform(np.column_stack([g.predict(Xn) for g in self.gps]))
        else:
            Yn = self.reg.predict(Xn)
        return Yn * self.sd + self.mu

    def predict_curves(self, params):
        """One set of 4 dials -> table of the 6 curves on the history dates."""
        out = pd.DataFrame(self.predict(params)[0].reshape(N_CURVES, N_T).T, columns=NAMES)
        out.insert(0, "Date", self.dates)
        return out


def score(Y_true, Y_pred):
    """Per-curve R2 (across cases), NRMSE %, and final-date % error for cumulative curves."""
    n = len(Y_true)
    T, P = Y_true.reshape(n, N_CURVES, N_T), Y_pred.reshape(n, N_CURVES, N_T)
    rows = []
    for k, name in enumerate(NAMES):
        t, p = T[:, k], P[:, k]
        r2 = 1 - ((t - p) ** 2).sum() / ((t - t.mean(0)) ** 2).sum()
        nrmse = np.sqrt(((t - p) ** 2).mean()) / (t.max() - t.min()) * 100
        final_err = np.mean(np.abs(t[:, -1] - p[:, -1]) / t[:, -1]) * 100 if k < 3 else np.nan
        rows.append({"Curve": name, "R2": r2, "NRMSE %": nrmse, "Final-date error %": final_err})
    return pd.DataFrame(rows).set_index("Curve")


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("proxy_outputs")
    out_dir.mkdir(exist_ok=True, parents=True)

    dates, X, Y, H, dial_names = load_data(data_dir)
    tr, va, te = slice(0, 70), slice(70, 85), slice(85, 100)
    print(f"Loaded {len(X)} cases | dials: {dial_names} | curves: {Y.shape[1]} values per case")

    # 1) Train 3 candidates on the training cases, compare on validation cases
    candidates = {k: Proxy(k).fit(X[tr], Y[tr]) for k in ["linear", "quadratic", "pca_gp"]}
    val_table = {}
    for k, m in candidates.items():
        s = score(Y[va], m.predict(X[va]))
        val_table[k] = {"mean R2": s["R2"].mean(), "mean NRMSE %": s["NRMSE %"].mean()}
    val_table = pd.DataFrame(val_table).T
    print("\nVALIDATION (cases 71-85)\n", val_table.round(4).to_string())
    best = val_table["mean NRMSE %"].idxmin()
    print(f"\nChosen on validation: {best}")

    # 2) Score the chosen model once on the unseen test cases
    test_pred = candidates[best].predict(X[te])
    test_scores = score(Y[te], test_pred)
    print("\nTEST (cases 86-100)\n", test_scores.round(4).to_string())

    # 3) Refit on train+validation (85 cases) for later use in history matching; save everything
    final = Proxy(best).fit(X[:85], Y[:85])
    final.dates, final.dial_names = dates, dial_names
    joblib.dump({"model": final, "bounds": (X.min(0), X.max(0)), "history": H}, out_dir / "proxy_model.joblib")
    val_table.to_csv(out_dir / "validation_scores.csv")
    test_scores.to_csv(out_dir / "test_scores.csv")

    # 4) Speed: how long to score 10,000 parameter sets?
    rng = np.random.default_rng(SEED)
    P = rng.uniform(X.min(0), X.max(0), (10000, X.shape[1]))
    t0 = time.time()
    final.predict(P)
    print(f"\nSpeed: {len(P)} parameter sets predicted in {time.time() - t0:.2f} s")

    # 5) Plots
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for k in range(3):
        t, p = Y[te].reshape(-1, N_CURVES, N_T)[:, k, -1], test_pred.reshape(-1, N_CURVES, N_T)[:, k, -1]
        ax[k].scatter(t, p)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax[k].plot(lim, lim, "k--", lw=1)
        ax[k].set(title=f"{NAMES[k]} at 2008 (test cases)", xlabel="Simulator", ylabel="Proxy")
    fig.tight_layout()
    fig.savefig(out_dir / "test_final_cumulative.png", dpi=150)

    T3 = Y[te].reshape(-1, N_CURVES, N_T)
    P3 = test_pred.reshape(-1, N_CURVES, N_T)
    worst = int(np.argmax(np.sqrt(((T3 - P3) / (T3.max(0) - T3.min(0) + 1e-9)) ** 2).mean(axis=(1, 2))))
    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    for k, a in enumerate(ax.ravel()):
        a.plot(dates, T3[worst, k], label="Simulator")
        a.plot(dates, P3[worst, k], "--", label="Proxy")
        a.set_title(NAMES[k])
    ax[0, 0].legend()
    fig.suptitle(f"Hardest test case: Case {86 + worst}")
    fig.tight_layout()
    fig.savefig(out_dir / "test_worst_case_curves.png", dpi=150)
    print(f"\nSaved model, scores and plots to {out_dir.resolve()}")


if __name__ == "__main__":
    import proxy_model  # run via the module name so the saved model can be re-loaded from other scripts

    proxy_model.main()
