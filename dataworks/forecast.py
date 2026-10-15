"""
DataWorks Challenge 2026 - Step 3: pick the best 10 cases, forecast 2008-2028, fill the template.

Run:   python forecast.py [data_folder] [template.xlsx] [output_folder]
Needs: proxy_model.py in the same folder (only for its load_data function), plus pandas, numpy, openpyxl, matplotlib.

What is different from the first version of this step (and why - every point was tested):
  1. RANKING: observed rate on date t = simulated rate on date t+1 (one-quarter shift). Cumulative lines up on the same
     date. We compare accordingly and include the cumulative curves. (Without the shift the wrong 10 cases get picked.)
  2. FORECAST MODEL is chosen by a backtest, not by guessing: hide the last 2-4 years, forecast them, keep the model
     with the lowest error on both the 100 cases and the real field. See backtest_grid.csv.
  3. The forecast starts from the REAL field's last values: observed cumulative, and the observed rate of the first
     forecast quarter (the observed rate on 1 Jan 2008 is the simulated rate for 1 Apr 2008).
  4. Alternative model structures are exported as scenarios so the dashboard can show the forecast uncertainty.
"""
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import openpyxl
import pandas as pd

from proxy_model import N_CURVES, N_T, load_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
N_BEST = 10
FLUIDS = ["oil", "gas", "water"]  # curve order in load_data: 3 cumulative, then 3 rates
BACKTEST_CUTS = ["2004-01-01", "2005-01-01", "2006-01-01"]  # hide everything after these dates
COLUMN_FOR = {  # template header -> (fluid index, kind)
    "Gas production cumulative [MSCF]": 1,
    "Oil production cumulative [STB]": 0,
    "Water production cumulative [STB]": 2,
}


# ------------------------------------------------------------------ ranking
def score_cases(Yy, Hh, c):
    """Mismatch of each case vs the observed history up to date index c. Recent years count more (weights 1 -> 4)."""
    s = Yy.std(0).max(1)  # one scale per curve type
    w = np.linspace(1, 4, c + 1)
    e_cum = ((Yy[:, :3, : c + 1] - Hh[:3, : c + 1]) / s[:3, None]) ** 2
    e_rate = ((Yy[:, 3:, 1 : c + 1] - Hh[3:, :c]) / s[3:, None]) ** 2  # observed t  <->  simulated t+1
    return np.sqrt((np.average(e_cum, axis=2, weights=w).mean(1) + np.average(e_rate, axis=2, weights=w[1:]).mean(1)) / 2)


# ------------------------------------------------------------------ forecast model
def extend(t_hist, q, t_fut, n_fit, liquid, water):
    """Extend (oil, gas, water) rates. Trend is fitted on the last n_fit quarters; times are years from the last date.
    liquid: 'exp' = exponential decline of oil+water | 'pow' = power-law decline (decline slows down over time)
    water : 'wor_exp' water/oil ratio grows exponentially | 'wor_lin' ratio grows linearly | 'fw_lin' water cut grows linearly (max 98%)
    gas = oil x recent gas/oil ratio."""
    qo, qg, qw = (x[-n_fit:] for x in q)
    t = t_hist[-n_fit:]
    ql = qo + qw
    if liquid == "exp":
        slope, icpt = np.polyfit(t, np.log(ql), 1)
        ql_f = np.exp(icpt + slope * t_fut)
    else:
        p, a = np.polyfit(np.log(t - t[0] + 1.0), np.log(ql), 1)
        ql_f = np.exp(a + p * np.log(t_fut - t[0] + 1.0))
    if water == "wor_exp":
        s2, b2 = np.polyfit(t, np.log(np.maximum(qw, 1e-9) / qo), 1)
        wor = np.exp(b2 + s2 * t_fut)
        fw = wor / (1 + wor)
    elif water == "wor_lin":
        s2, b2 = np.polyfit(t, qw / qo, 1)
        wor = np.maximum(b2 + s2 * t_fut, 0)
        fw = wor / (1 + wor)
    else:
        s2, b2 = np.polyfit(t, qw / ql, 1)
        fw = np.clip(b2 + s2 * t_fut, 0, 0.98)
    qo_f = ql_f * (1 - fw)
    return np.array([qo_f, np.mean(qg / qo) * qo_f, ql_f * fw])


def forecast_rates(q, t_hist, t_fut, cfg, next_rate):
    """Extend rates, then shift them so the first forecast quarter equals the known rate (next_rate)."""
    f = extend(t_hist, q, t_fut, *cfg)
    return f * (next_rate / f[:, 0])[:, None]


def integrate(c0, q_start, f, t_fut):
    """Cumulative = start value + trapezoid sum of rates x days (this matched the simulator's own cumulative to <1%)."""
    days = np.diff(np.r_[0.0, t_fut]) * 365.25
    out = np.zeros_like(f)
    for k in range(3):
        qq = np.r_[q_start[k], f[k]]
        out[k] = c0[k] + np.cumsum(0.5 * (qq[:-1] + qq[1:]) * days)
    return out


# ------------------------------------------------------------------ backtest to choose the model
def backtest(Yy, Hh, dates, t_h, cfg):
    """Mean |error| of the forecast production over the hidden period: (100 cases, real field)."""
    obs_sim = np.zeros((3, N_T))  # observed rates written in simulator convention
    obs_sim[:, 1:] = Hh[3:6, :-1]
    case_err, real_err = [], []
    for cut in BACKTEST_CUTS:
        c = int(np.where(dates.values == np.datetime64(cut))[0][0])
        th, tf = t_h[: c + 1] - t_h[c], t_h[c + 1 :] - t_h[c]

        def hidden_error(q, nxt, cum_end, cum_cut):
            f = forecast_rates(q[:, : c + 1], th, tf, cfg, nxt)
            pred = integrate(np.zeros(3), q[:, c], f, tf)[:, -1]
            return pred / (cum_end - cum_cut) - 1

        for i in range(100):
            case_err.append(hidden_error(Yy[i, 3:6], Yy[i, 3:6, c + 1], Yy[i, :3, -1], Yy[i, :3, c]))
        real_err.append(hidden_error(obs_sim, Hh[3:6, c], Hh[:3, -1], Hh[:3, c]))
    return np.abs(case_err).mean() * 100, np.abs(real_err).mean() * 100, np.mean(case_err, 0) * 100, np.mean(real_err, 0) * 100


def parse_date(v):
    return v if isinstance(v, datetime) else datetime.strptime(str(v).strip(), "%m/%d/%Y")


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    template = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "09_Template_Deliverable.xlsx"
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("forecast_outputs")
    out_dir.mkdir(exist_ok=True, parents=True)

    dates, X, Y, H, _ = load_data(data_dir)
    Yy, Hh = Y.reshape(100, N_CURVES, N_T), H.reshape(N_CURVES, N_T)
    t_h = (dates - dates.iloc[-1]).dt.days.values / 365.25

    # 1) choose the forecast model by backtest (lowest average of case error and real-field error)
    # A short backtest cannot see 20 years ahead, so we also reject any model whose water cut hits its 98% ceiling
    # within the forecast horizon: the oil rate would then be forced onto a floor by the ceiling, not by the data.
    obs_sim = np.zeros((3, N_T))
    obs_sim[:, 1:] = Hh[3:6, :-1]
    t_check = np.arange(0.25, 20.01, 0.25)
    grid = []
    for liquid, water, n in product(["exp", "pow"], ["wor_exp", "wor_lin", "fw_lin"], [6, 8, 12]):
        ce, re, cb, rb = backtest(Yy, Hh, dates, t_h, (n, liquid, water))
        f = extend(t_h, obs_sim, t_check, n, liquid, water)
        capped = bool((f[2] / (f[0] + f[2])).max() >= 0.979)
        grid.append({"liquid": liquid, "water": water, "window": n, "cases error %": ce, "real field error %": re,
                     "combined": (ce + re) / 2, "cases oil bias %": cb[0], "cases water bias %": cb[2], "hits 98% water-cut ceiling": capped})
    grid = pd.DataFrame(grid).sort_values("combined").round(2)
    grid.to_csv(out_dir / "backtest_grid.csv", index=False)
    usable = grid[~grid["hits 98% water-cut ceiling"]]
    print("Backtest (hide last 2-4 years, forecast them). Best models overall by combined error %:\n", grid.head(4).to_string(index=False))
    print("\nBest models WITHOUT the 98% ceiling problem (these are the ones we may use):\n", usable.head(4).to_string(index=False))
    cfgs = [(int(r["window"]), r["liquid"], r["water"]) for _, r in usable.iterrows()]
    primary = cfgs[0]
    print(f"\nChosen: liquid={primary[1]}, water={primary[2]}, window={primary[0]} quarters")
    scenarios = {"Primary": primary}
    for label, cfg in zip(["Alt 1", "Alt 2", "Original method"], [cfgs[1], cfgs[2], (8, "exp", "wor_exp")]):
        if cfg not in scenarios.values():
            scenarios[label] = cfg

    # 2) pick the best 10 existing cases (shift-aware score using ALL history)
    c_end = N_T - 1
    score = pd.Series(score_cases(Yy, Hh, c_end), index=[f"Case_{i}" for i in range(1, 101)]).sort_values()
    top = [int(n.split("_")[1]) - 1 for n in score.index[:N_BEST]]
    print("\nBest 10 cases:", list(score.index[:N_BEST]))

    # 3) template dates and forecast (starting from the REAL field's last values)
    wb = openpyxl.load_workbook(template)
    tdates = [parse_date(r[0]) for r in wb["Case 1"].iter_rows(min_row=2, max_col=1, values_only=True) if r[0] is not None]
    assert len(tdates) == 80 and tdates[0] == datetime(2008, 4, 1) and tdates[-1] == datetime(2028, 1, 1), "unexpected template dates"
    future = pd.DatetimeIndex(tdates)
    t_fut = (future - dates.iloc[-1]).days.values / 365.25
    c0 = Hh[:3, -1]  # observed cumulative oil, gas, water at 1 Jan 2008
    q_start = Hh[3:6, -2]  # rate over the last history quarter (simulator convention)
    next_rate = Hh[3:6, -1]  # observed rate on 1 Jan 2008 = first forecast quarter

    rows, finals = [], {}
    obs_rate_sim = np.zeros((3, N_T))
    obs_rate_sim[:, 1:] = Hh[3:6, :-1]
    for rank, i in enumerate(top, start=1):
        for name, cfg in scenarios.items():
            f = forecast_rates(Yy[i, 3:6], t_h, t_fut, cfg, next_rate)
            cum = integrate(c0, q_start, f, t_fut)
            finals[(name, rank)] = cum[:, -1]
            df = pd.DataFrame({"Date": future, "Oil cumulative [STB]": cum[0], "Gas cumulative [MSCF]": cum[1],
                               "Water cumulative [STB]": cum[2], "Oil rate [STB/d]": f[0], "Gas rate [MSCF/d]": f[1],
                               "Water rate [STB/d]": f[2]})
            df.insert(0, "Rank", rank)
            df.insert(0, "Case", f"Case_{i + 1}")
            df.insert(0, "Scenario", name)
            rows.append(df)
            if name == "Primary":  # fill template sheet "Case <rank>"
                ws = wb[f"Case {rank}"]
                header = [cell.value for cell in ws[1]]
                for col, h in enumerate(header, start=1):
                    if h in COLUMN_FOR:
                        for r, v in enumerate(cum[COLUMN_FOR[h]], start=2):
                            ws.cell(row=r, column=col, value=float(v))
    out_xlsx = out_dir / "09_Template_Deliverable_filled.xlsx"
    wb.save(out_xlsx)

    hist = pd.DataFrame({"Scenario": "Observed history", "Case": "Observed", "Rank": 0, "Date": dates,
                         "Oil cumulative [STB]": Hh[0], "Gas cumulative [MSCF]": Hh[1], "Water cumulative [STB]": Hh[2],
                         "Oil rate [STB/d]": obs_rate_sim[0], "Gas rate [MSCF/d]": obs_rate_sim[1], "Water rate [STB/d]": obs_rate_sim[2]})
    pd.concat([hist] + rows).to_csv(out_dir / "forecast_all_cases_for_powerbi.csv", index=False)
    pd.DataFrame({"Template sheet": [f"Case {r}" for r in range(1, N_BEST + 1)], "Simulation case": list(score.index[:N_BEST]),
                  "Mismatch score": score.values[:N_BEST]}).to_csv(out_dir / "best10_summary.csv", index=False)

    # 4) final checks on the file we just wrote
    chk = openpyxl.load_workbook(out_xlsx, data_only=True)
    for r in range(1, N_BEST + 1):
        vals = np.array([[v for v in row[1:4]] for row in chk[f"Case {r}"].iter_rows(min_row=2, max_row=81, values_only=True)], dtype=float)
        assert vals.shape == (80, 3) and not np.isnan(vals).any(), f"Case {r}: blank or wrong size"
        assert (np.diff(vals, axis=0) >= 0).all(), f"Case {r}: cumulative decreases"
        assert (vals[0] >= Hh[[1, 0, 2], -1]).all(), f"Case {r}: forecast starts below the 2008 history"
    print("\nFile checks passed: 10 sheets x 80 rows, no blanks, cumulative never decreases, starts at the 2008 history.")
    print("\n2028 cumulative production range across the 10 cases, per scenario (millions):")
    for name in scenarios:
        a = np.array([finals[(name, r)] for r in range(1, N_BEST + 1)]) / 1e6
        print(f"  {name:24s} oil {a[:, 0].min():5.1f}-{a[:, 0].max():5.1f} | gas {a[:, 1].min():5.1f}-{a[:, 1].max():5.1f} | water {a[:, 2].min():5.1f}-{a[:, 2].max():5.1f}")

    # 5) picture: observed history, primary forecasts, and the range across alternative model structures
    long = pd.concat(rows)
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for k, (cum_col, rate_col, title) in enumerate(zip(["Oil cumulative [STB]", "Gas cumulative [MSCF]", "Water cumulative [STB]"],
                                                       ["Oil rate [STB/d]", "Gas rate [MSCF/d]", "Water rate [STB/d]"], FLUIDS)):
        for row, col in enumerate([cum_col, rate_col]):
            a = ax[row, k]
            piv = long.pivot_table(index="Date", columns="Scenario", values=col, aggfunc=["min", "max"])
            lo_ = piv["min"].min(axis=1)
            hi_ = piv["max"].max(axis=1)
            a.fill_between(piv.index, lo_, hi_, color="lightblue", alpha=0.5, label="range across model structures")
            for r in range(1, N_BEST + 1):
                d = long[(long.Scenario == "Primary") & (long.Rank == r)]
                a.plot(d.Date, d[col], lw=0.8)
            a.plot(hist.Date, hist[col], "k", lw=2, label="Observed history")
            a.set_title(col)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "forecast.png", dpi=150)
    print(f"\nSaved to {out_dir.resolve()}")


if __name__ == "__main__":
    import forecast

    forecast.main()
