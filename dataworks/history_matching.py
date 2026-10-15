"""
DataWorks Challenge 2026 - Step 2: History matching
Find the 4 dial settings whose proxy curves best match the REAL field history (RAMP_History),
then pick 10 settings that all fit well but are deliberately different from each other.

Run:   python history_matching.py [data_folder] [proxy_folder] [output_folder]
Needs: proxy_model.py in the same folder, and proxy_results/proxy_model.joblib from Step 1.

DATA QUIRK (important): in the observed history the oil/gas RATE on a date equals the simulated rate
one quarter LATER (obs 1 Oct 1998 = simulated 1 Jan 1999). Cumulative curves line up on the same date.
We therefore compare observed_rate[t] with simulated_rate[t+1]. Matching rates without this shift gives nonsense.
"""
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from proxy_model import NAMES, N_CURVES, N_T, Proxy, load_data  # noqa: F401  (Proxy needed to load the pickle)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 0
N_SETS = 10          # how many final dial settings to deliver
N_SEARCH = 200_000   # random dial settings tried through the proxy
ACCEPT_MARGIN = 2.0  # accept settings within (best mismatch + 2 x proxy error floor)


def score_curves(pred, target, sd):
    """Mismatch (scaled RMS error) between predicted curves (n x 246) and one observed history (246)."""
    n = len(pred)
    scale = sd.reshape(N_CURVES, N_T)[:, 0]  # one scale per curve type so no fluid dominates
    p, t = pred.reshape(n, N_CURVES, N_T), target.reshape(N_CURVES, N_T)
    e_cum = (p[:, :3] - t[:3]) / scale[:3, None]
    e_rate = (p[:, 3:, 1:] - t[3:, :-1]) / scale[3:, None]  # observed rate at t  <->  simulated rate at t+1
    errors = np.concatenate([e_cum.reshape(n, -1), e_rate.reshape(n, -1)], axis=1)
    return np.sqrt((errors ** 2).mean(1))


def as_observed(case_curves):
    """Write a simulated case in the 'observed' convention (rates shifted one quarter earlier)."""
    h = case_curves.reshape(N_CURVES, N_T).copy()
    h[3:, :-1] = h[3:, 1:]
    return h.ravel()


def make_mismatch(model, target):
    return lambda P: score_curves(model.predict(P), target, model.sd)


def find_best(mismatch, lo, hi):
    r = differential_evolution(lambda x: mismatch(x.T), list(zip(lo, hi)), seed=SEED, tol=1e-10,
                               maxiter=300, vectorized=True, updating="deferred", polish=False)
    return r.x, float(r.fun)


def pick_diverse(Z, f, k):
    """Start from the best fit, then repeatedly add the accepted setting farthest from those already chosen."""
    chosen = [int(np.argmin(f))]
    d = np.linalg.norm(Z - Z[chosen[0]], axis=1)
    while len(chosen) < min(k, len(Z)):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(Z - Z[i], axis=1))
    return chosen


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    proxy_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("proxy_results")
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("matching_outputs")
    out_dir.mkdir(exist_ok=True, parents=True)

    bundle = joblib.load(proxy_dir / "proxy_model.joblib")
    model, H, (lo, hi) = bundle["model"], bundle["history"], bundle["bounds"]
    dates, X, Y, _, dial_names = load_data(data_dir)
    mismatch = make_mismatch(model, H)

    # 1) Reference points: best of the 100 given cases, and the proxy's own error level
    given = score_curves(Y, H, model.sd)
    floor = np.median([make_mismatch(model, as_observed(Y[i]))(X[i:i + 1])[0] for i in range(85, 100)])
    print(f"Best given case: Case {given.argmin() + 1} (mismatch {given.min():.4f})")
    print(f"Proxy error floor on unseen test cases (median mismatch): {floor:.4f}")

    # 2) Search: many space-filling random dial settings + an optimiser for the exact best
    P = qmc.scale(qmc.LatinHypercube(d=4, seed=SEED).random(N_SEARCH), lo, hi)
    f = np.concatenate([mismatch(P[i:i + 20000]) for i in range(0, N_SEARCH, 20000)])
    best_x, best_f = find_best(mismatch, lo, hi)
    P, f = np.vstack([P, best_x]), np.append(f, best_f)
    print(f"Best match found: mismatch {best_f:.4f}")
    print("   at", dict(zip(dial_names, np.round(best_x, 3))))
    edge = [n for n, x, a, b in zip(dial_names, best_x, lo, hi) if min(x - a, b - x) < 0.01 * (b - a)]
    if edge:
        print("   WARNING - best answer sits on the allowed limit for:", edge)

    # 3) Which settings fit "about as well as the best"? How well does history pin down each dial?
    ok = f <= best_f + ACCEPT_MARGIN * floor
    width = (P[ok].max(0) - P[ok].min(0)) / (hi - lo) * 100
    print(f"\nAcceptable settings: {ok.sum()} of {len(f)} tried (mismatch <= {best_f + ACCEPT_MARGIN * floor:.4f})")

    # 4) Recovery test: can we find the true dials of cases the proxy never saw (86-100)?
    rec = []
    for i in range(85, 100):
        x, _ = find_best(make_mismatch(model, as_observed(Y[i])), lo, hi)
        rec.append(np.abs(x - X[i]) / (hi - lo) * 100)
    rec = np.mean(rec, axis=0)

    ident = pd.DataFrame({"Dial": dial_names, "Allowed min": lo, "Allowed max": hi,
                          "Acceptable range width (% of allowed)": width,
                          "Recovery error on unseen cases (% of allowed)": rec}).round(2)
    ident.to_csv(out_dir / "dial_identifiability.csv", index=False)
    print("\nHOW WELL DOES HISTORY PIN DOWN EACH DIAL?  (small = well determined, ~100 = not at all)\n",
          ident.drop(columns=["Allowed min", "Allowed max"]).to_string(index=False))

    # 5) Choose the final 10: good fits that are spread out
    Pok, fok = P[ok], f[ok]
    idx = pick_diverse((Pok - lo) / (hi - lo), fok, N_SETS)
    final = pd.DataFrame(Pok[idx], columns=dial_names)
    final.insert(0, "Set", range(1, len(idx) + 1))
    final["Mismatch"] = fok[idx]
    final["Oil cum 2008"] = [model.predict(p)[0].reshape(N_CURVES, N_T)[0, -1] for p in Pok[idx]]
    final["Gas cum 2008"] = [model.predict(p)[0].reshape(N_CURVES, N_T)[1, -1] for p in Pok[idx]]
    final["Water cum 2008"] = [model.predict(p)[0].reshape(N_CURVES, N_T)[2, -1] for p in Pok[idx]]
    final.round(4).to_csv(out_dir / "matched_dials.csv", index=False)
    print("\nFINAL", len(idx), "SETS\n", final.round(3).to_string(index=False))

    curves = []
    for s, p in zip(final["Set"], Pok[idx]):
        c = model.predict_curves(p)
        c.insert(0, "Set", s)
        curves.append(c)
    curves = pd.concat(curves)
    curves.to_csv(out_dir / "matched_curves.csv", index=False)

    # 6) Plot: 100 given cases (grey), observed history (black), the 10 matched sets (colour)
    Yg, Hh = Y.reshape(-1, N_CURVES, N_T), H.reshape(N_CURVES, N_T)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for k, a in enumerate(ax.ravel()):
        for case in Yg:
            a.plot(dates, case[k], color="lightgrey", lw=0.6)
        for s in final["Set"]:
            a.plot(dates, curves[curves.Set == s][NAMES[k]].values, lw=1.2)
        if k < 3:
            a.plot(dates, Hh[k], "k.", ms=4)
        else:  # shift observed rates one quarter so they sit on the dates they correspond to
            a.plot(dates[1:], Hh[k][:-1], "k.", ms=4)
        a.set_title(NAMES[k])
    fig.suptitle("Observed history (black dots) vs 100 given cases (grey) vs 10 matched sets (colour)")
    fig.tight_layout()
    fig.savefig(out_dir / "history_match.png", dpi=150)
    print(f"\nSaved results to {out_dir.resolve()}")


if __name__ == "__main__":
    import history_matching

    history_matching.main()
