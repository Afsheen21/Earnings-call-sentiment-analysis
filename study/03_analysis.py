"""
STEP 3 - Test the hypotheses in HYPOTHESES.md. Prints results and saves tables + figures to results/.

Run from the ROOT of your project folder AFTER steps 1 and 2:
    python study/03_analysis.py

What it does (and why):
 A. Spearman correlation of each sentiment score with the move size (abs_ret) and direction
    (signed_ret), with bootstrap 95% confidence intervals.
 B. Permutation test: shuffle scores WITHIN each company 5000 times. If the real correlation is
    not larger than the shuffled ones, it could be chance. Holm-corrected for multiple tests.
 C. Regression: does sentiment explain abs_ret AFTER controlling for pre-event volatility?
    Standard errors are clustered by company (calls from one company are not independent).
 D. Out-of-sample test: train on the earliest 70% of calls, test on the latest 30%.
    Does adding sentiment beat "pre-event volatility alone"? (paired bootstrap CI)
 E. H1: are the finance-specific scorers (LM, FinBERT) more informative than VADER?
 F. H3: is the Q&A section more informative than the prepared remarks?
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)          # fixed seed so your results are reproducible
N_BOOT, N_PERM = 2000, 5000
os.makedirs("results", exist_ok=True)

ev = pd.read_csv("data/events.csv", parse_dates=["t0"])
sc = pd.read_csv("data/sentiment_scores.csv")
d = ev.merge(sc, on="doc_id").sort_values("t0").reset_index(drop=True)
METHODS = [m for m in ["vader", "lm", "finbert"] if f"{m}_all" in d.columns]
print(f"Calls analysed: {len(d)} | companies: {d['ticker'].nunique()} | methods: {METHODS}\n")


def ols(X, y):
    """Plain least squares with an intercept. Returns coefficients (intercept first) and R^2."""
    X1 = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    beta = np.linalg.lstsq(X1, np.asarray(y, float), rcond=None)[0]
    resid = y - X1 @ beta
    r2 = 1 - (resid ** 2).sum() / ((y - np.mean(y)) ** 2).sum()
    return beta, r2


def predict(beta, X):
    return np.column_stack([np.ones(len(X)), np.asarray(X, float)]) @ beta


def cluster_p(X, y, groups, j):
    """Cluster-robust p-value for coefficient j (j=1 is the first non-intercept column).
    Calls from the same company are not independent, so errors are clustered by company."""
    X1 = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    e = y - X1 @ beta
    bread = np.linalg.inv(X1.T @ X1)
    meat = np.zeros((X1.shape[1], X1.shape[1]))
    g_ids = pd.unique(groups)
    for g in g_ids:
        m = np.asarray(groups) == g
        u = X1[m].T @ e[m]
        meat += np.outer(u, u)
    G, N, K = len(g_ids), len(y), X1.shape[1]
    V = bread @ meat @ bread * (G / (G - 1)) * ((N - 1) / (N - K))
    t = beta[j] / np.sqrt(V[j, j])
    return beta[j], 2 * (1 - stats.t.cdf(abs(t), df=G - 1))


def holm(pvals):
    """Holm step-down correction for multiple tests."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rho(x, y):
    return spearmanr(x, y)[0]


def boot_ci(x, y, n=N_BOOT):
    idx = np.arange(len(x))
    vals = [rho(x[i], y[i]) for i in (RNG.choice(idx, len(idx)) for _ in range(n))]
    return np.nanpercentile(vals, [2.5, 97.5])


# ---------- A + B: correlations, bootstrap CIs, within-company permutation test ----------
rows = []
for m in METHODS:
    col = f"{m}_all"
    x = d[col].values
    for target in ["abs_ret", "signed_ret"]:
        y = d[target].values
        obs = rho(x, y)
        lo, hi = boot_ci(x, y)
        perm = []
        for _ in range(N_PERM):
            xp = x.copy()
            for _, idx in d.groupby("ticker").indices.items():
                xp[idx] = RNG.permutation(xp[idx])
            perm.append(abs(rho(xp, y)))
        p = (1 + np.sum(np.array(perm) >= abs(obs))) / (1 + N_PERM)
        rows.append(dict(method=m, target=target, spearman_rho=obs, ci_low=lo, ci_high=hi, perm_p=p))
corr = pd.DataFrame(rows)
corr["p_holm"] = holm(corr["perm_p"])
corr.round(4).to_csv("results/A_correlations.csv", index=False)
print("A/B. Correlations (Spearman) with 95% bootstrap CI, permutation p, Holm-corrected p:")
print(corr.round(3).to_string(index=False), "\n")

# ---------- C: regression controlling for pre-event volatility ----------
reg_rows = []
for m in METHODS:
    z = (d[f"{m}_all"] - d[f"{m}_all"].mean()) / d[f"{m}_all"].std()      # z-scored sentiment
    X = np.column_stack([z, d["pre_vol"]])
    ok = ~np.isnan(X).any(axis=1)
    _, r2_base = ols(d.loc[ok, ["pre_vol"]].values, d.loc[ok, "abs_ret"].values)
    _, r2_full = ols(X[ok], d.loc[ok, "abs_ret"].values)
    coef, p = cluster_p(X[ok], d.loc[ok, "abs_ret"].values, d.loc[ok, "ticker"].values, 1)
    reg_rows.append(dict(method=m, coef_sentiment_z=coef, p_value=p,
                         r2_baseline=r2_base, r2_with_sentiment=r2_full))
reg = pd.DataFrame(reg_rows)
reg.round(4).to_csv("results/C_regression.csv", index=False)
print("C. abs_ret ~ pre_vol + sentiment (cluster-robust by company):")
print(reg.round(4).to_string(index=False), "\n")

# ---------- D: out-of-sample, time-ordered ----------
cut = int(0.7 * len(d))
tr, te = d.iloc[:cut], d.iloc[cut:]


oos_rows = []
for m in METHODS:
    c = f"{m}_all"
    t_tr, t_te = tr.dropna(subset=[c]), te.dropna(subset=[c])
    b0, _ = ols(t_tr[["pre_vol"]].values, t_tr["abs_ret"].values)
    b1, _ = ols(t_tr[["pre_vol", c]].values, t_tr["abs_ret"].values)
    err0 = (t_te["abs_ret"].values - predict(b0, t_te[["pre_vol"]].values)) ** 2
    err1 = (t_te["abs_ret"].values - predict(b1, t_te[["pre_vol", c]].values)) ** 2
    diff = err0 - err1                                   # > 0 means sentiment helped on that call
    boots = [diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
    oos_rows.append(dict(method=m, n_test=len(diff), mse_baseline=err0.mean(),
                         mse_with_sentiment=err1.mean(), mean_improvement=diff.mean(),
                         ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5)))
oos = pd.DataFrame(oos_rows)
oos.round(4).to_csv("results/D_out_of_sample.csv", index=False)
print(f"D. Out-of-sample (train {len(tr)} earliest calls, test {len(te)} latest). "
      "mean_improvement > 0 and CI above 0 means sentiment helps:")
print(oos.round(4).to_string(index=False), "\n")

# ---------- E: H1 finance-specific vs VADER ----------
if "vader" in METHODS:
    h1 = []
    for m in [x for x in METHODS if x != "vader"]:
        idx = np.arange(len(d)); diffs = []
        for _ in range(N_BOOT):
            i = RNG.choice(idx, len(idx))
            diffs.append(abs(rho(d[f"{m}_all"].values[i], d["abs_ret"].values[i])) -
                         abs(rho(d["vader_all"].values[i], d["abs_ret"].values[i])))
        obs = abs(rho(d[f"{m}_all"], d["abs_ret"])) - abs(rho(d["vader_all"], d["abs_ret"]))
        h1.append(dict(method=m, abs_rho_minus_vader=obs, ci_low=np.percentile(diffs, 2.5),
                       ci_high=np.percentile(diffs, 97.5)))
    h1 = pd.DataFrame(h1)
    h1.round(4).to_csv("results/E_H1_vs_vader.csv", index=False)
    print("E. H1: |rho| with abs_ret, method minus VADER (CI above 0 supports H1):")
    print(h1.round(4).to_string(index=False), "\n")

# ---------- F: H3 prepared remarks vs Q&A ----------
f_rows = []
for m in METHODS:
    for part in ["prep", "qa"]:
        sub = d.dropna(subset=[f"{m}_{part}"])
        if len(sub) > 10:
            f_rows.append(dict(method=m, part=part, n=len(sub),
                               spearman_rho_abs_ret=rho(sub[f"{m}_{part}"], sub["abs_ret"])))
sec = pd.DataFrame(f_rows)
sec.round(4).to_csv("results/F_H3_sections.csv", index=False)
print("F. H3: prepared remarks vs Q&A (Spearman with abs_ret):")
print(sec.round(4).to_string(index=False), "\n")

# ---------- Figure ----------
sub = corr[corr["target"] == "abs_ret"]
plt.figure(figsize=(6, 4))
plt.errorbar(sub["method"], sub["spearman_rho"],
             yerr=[sub["spearman_rho"] - sub["ci_low"], sub["ci_high"] - sub["spearman_rho"]],
             fmt="o", capsize=5)
plt.axhline(0, color="gray", ls="--", lw=0.8)
plt.ylabel("Spearman rho with |return| (95% CI)")
plt.title("Sentiment vs size of the post-earnings move")
plt.tight_layout(); plt.savefig("results/fig_correlations.png", dpi=150)
print("Saved tables and figure in results/. Now read them honestly - see the README steps.")
