"""
STEP 5 - Robustness checks. Run AFTER 03_analysis.py:
    python study/05_robustness.py

WHY: your first results show a large in-sample correlation (LM rho = 0.38) but NO out-of-sample gain.
A result like that is exactly what happens when something else drives both numbers. These checks ask:
  R1. Does the correlation survive controlling for pre-event volatility AND the year (market regime)?
      (variant b also removes company effects with company dummies)
  R2. Is it driven by one company? (drop each company in turn)
  R3. Is it driven by a few huge moves? (drop the top 5% of moves)
  R4. H3 with a confidence interval: is Q&A really better than prepared remarks, or is that noise?
  R5. Is the out-of-sample 'no gain' sensitive to the split (60/40, 70/30, 80/20) or the target
      (raw move vs log move)?
Report ALL of it, whichever way it goes. Saved to results/robustness_*.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

RNG = np.random.default_rng(42)
N_BOOT = 2000
os.makedirs("results", exist_ok=True)

ev = pd.read_csv("data/events.csv", parse_dates=["t0"])
sc = pd.read_csv("data/sentiment_scores.csv")
d = ev.merge(sc, on="doc_id").sort_values("t0").reset_index(drop=True)
d["year"] = d["t0"].dt.year
METHODS = [m for m in ["vader", "lm", "finbert"] if f"{m}_all" in d.columns]
print(f"Calls: {len(d)} | companies: {d['ticker'].nunique()} | years: {d['year'].min()}-{d['year'].max()}\n")


def partial_rho(df, xcol, ycol="abs_ret", dummies=("year",), with_ticker=False):
    """Spearman correlation of x and y AFTER removing what pre_vol (ranked) and dummies explain."""
    df = df.dropna(subset=[xcol, ycol, "pre_vol"])
    ctrl = [rankdata(df["pre_vol"])]
    cols = [np.ones(len(df)), ctrl[0]]
    for dcol in list(dummies) + (["ticker"] if with_ticker else []):
        dm = pd.get_dummies(df[dcol], drop_first=True).astype(float)
        cols += [dm[c].values for c in dm.columns]
    Z = np.column_stack(cols)
    res = []
    for col in (xcol, ycol):
        r = rankdata(df[col])
        beta = np.linalg.lstsq(Z, r, rcond=None)[0]
        res.append(r - Z @ beta)
    return float(np.corrcoef(res[0], res[1])[0, 1])


def boot(fn, df, n=N_BOOT):
    vals = []
    for _ in range(n):
        s = df.iloc[RNG.integers(0, len(df), len(df))]
        try:
            vals.append(fn(s))
        except Exception:
            pass
    return np.nanpercentile(vals, [2.5, 97.5])


# ---------- R1: partial correlations ----------
r1 = []
for m in METHODS:
    c = f"{m}_all"
    base = float(spearmanr(d[c], d["abs_ret"], nan_policy="omit")[0])
    pa = partial_rho(d, c)                                    # control pre_vol + year
    pb = partial_rho(d, c, with_ticker=True)                  # + company dummies
    lo, hi = boot(lambda s: partial_rho(s, c), d)
    lo2, hi2 = boot(lambda s: partial_rho(s, c, with_ticker=True), d)
    r1.append(dict(method=m, raw_rho=base, rho_ctrl_prevol_year=pa, ci_a=f"[{lo:.3f}, {hi:.3f}]",
                   rho_ctrl_prevol_year_company=pb, ci_b=f"[{lo2:.3f}, {hi2:.3f}]"))
r1 = pd.DataFrame(r1)
print("R1. Spearman with abs_ret after controlling for pre-event volatility + year (+ company):")
print(r1.round(3).to_string(index=False), "\n")
r1.to_csv("results/robustness_R1_partial.csv", index=False)

# ---------- R2: leave-one-company-out ----------
r2 = []
for m in METHODS:
    c = f"{m}_all"
    vals = {t: float(spearmanr(g[c], g["abs_ret"], nan_policy="omit")[0])
            for t, g in ((t, d[d.ticker != t]) for t in d.ticker.unique())}
    worst = min(vals, key=vals.get)
    r2.append(dict(method=m, min_rho=min(vals.values()), max_rho=max(vals.values()),
                   company_whose_removal_lowers_rho_most=worst))
r2 = pd.DataFrame(r2)
print("R2. Leave-one-company-out (range of rho with abs_ret):")
print(r2.round(3).to_string(index=False), "\n")
r2.to_csv("results/robustness_R2_leave_one_out.csv", index=False)

# ---------- R3: drop the biggest 5% of moves ----------
cut = d["abs_ret"].quantile(0.95)
dt = d[d["abs_ret"] <= cut]
r3 = pd.DataFrame([dict(method=m, n=len(dt), rho_without_top5pct=float(
    spearmanr(dt[f"{m}_all"], dt["abs_ret"], nan_policy="omit")[0])) for m in METHODS])
print(f"R3. Without the top 5% of moves (abs_ret > {cut:.2f}%):")
print(r3.round(3).to_string(index=False), "\n")
r3.to_csv("results/robustness_R3_no_outliers.csv", index=False)

# ---------- R4: H3 with a confidence interval ----------
r4 = []
for m in METHODS:
    q, p = f"{m}_qa", f"{m}_prep"
    if q not in d or p not in d:
        continue
    s = d.dropna(subset=[q, p])

    def diff(df):
        return spearmanr(df[q], df["abs_ret"])[0] - spearmanr(df[p], df["abs_ret"])[0]
    lo, hi = boot(diff, s)
    r4.append(dict(method=m, n=len(s), rho_qa=spearmanr(s[q], s["abs_ret"])[0],
                   rho_prep=spearmanr(s[p], s["abs_ret"])[0], diff_qa_minus_prep=diff(s),
                   ci=f"[{lo:.3f}, {hi:.3f}]", H3_supported=bool(lo > 0)))
r4 = pd.DataFrame(r4)
print("R4. H3 - Q&A minus prepared remarks (supported only if the CI is entirely above 0):")
print(r4.round(3).to_string(index=False), "\n")
r4.to_csv("results/robustness_R4_H3.csv", index=False)


# ---------- R5: out-of-sample under different splits and targets ----------
def ols_fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def ols_pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


r5 = []
for target in ("abs_ret", "log_abs_ret"):
    d["log_abs_ret"] = np.log(d["abs_ret"] + 0.1)
    for frac in (0.6, 0.7, 0.8):
        k = int(len(d) * frac)
        tr, te = d.iloc[:k], d.iloc[k:]
        for m in METHODS:
            c = f"{m}_all"
            a, b = tr.dropna(subset=[c]), te.dropna(subset=[c])
            b0 = ols_fit(a[["pre_vol"]].values, a[target].values)
            b1 = ols_fit(a[["pre_vol", c]].values, a[target].values)
            e0 = (b[target].values - ols_pred(b0, b[["pre_vol"]].values)) ** 2
            e1 = (b[target].values - ols_pred(b1, b[["pre_vol", c]].values)) ** 2
            diff = e0 - e1
            bs = [diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            r5.append(dict(target=target, train_frac=frac, method=m, n_test=len(diff),
                           mean_improvement=diff.mean(), ci=f"[{lo:.3f}, {hi:.3f}]", sentiment_helps=bool(lo > 0)))
r5 = pd.DataFrame(r5)
print("R5. Out-of-sample: does adding sentiment beat pre_vol alone? (helps only if CI above 0)")
print(r5.round(3).to_string(index=False), "\n")
r5.to_csv("results/robustness_R5_oos.csv", index=False)

print("Now read ALL of it together. Report what survives and what does not.")
