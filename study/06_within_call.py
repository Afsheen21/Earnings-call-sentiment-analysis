"""
STEP 6 - Does tone CHANGE inside a call carry information that one averaged score hides?
Tests the hypotheses in HYPOTHESES_v2.md. Commit that file to GitHub BEFORE you run this.

Run from the ROOT of your project folder (needs data/events.csv and data/sentiment_scores.csv from steps 1-2):
    python study/06_within_call.py

Only the Loughran-McDonald dictionary is used (it is fast; no FinBERT, so this takes about a minute).
Features
  d_qa_prep   : LM tone of Q&A  minus  LM tone of prepared remarks
  d_qa_halves : LM tone of the 2nd half of the Q&A  minus  the 1st half (halves split by sentence count)
Tests (same methods as steps 3 and 5)
  A. Spearman with abs_ret and signed_ret, bootstrap CI, within-company permutation p, Holm over 4 tests
  B. Partial Spearman with abs_ret after controlling pre_vol + year + company
  C. Out-of-sample: does pre_vol + feature beat pre_vol alone? (time-ordered 70/30, paired bootstrap)
Saves results/G_within_call.csv, results/G_within_call_oos.csv
"""
import importlib.util
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

RNG = np.random.default_rng(42)
N_BOOT, N_PERM = 2000, 5000
os.makedirs("results", exist_ok=True)

# reuse the Q&A splitter from step 2 (file name starts with a digit, so load it by path)
spec = importlib.util.spec_from_file_location("score02", os.path.join(os.path.dirname(__file__), "02_score_sentiment.py"))
score02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score02)
split_sections, sentences = score02.split_sections, score02.sentences

import pysentiment2 as ps
lm = ps.LM()


def tone(text):
    """LM net tone, or NaN if the text is empty or has no LM words."""
    if not text or not text.strip():
        return np.nan
    s = lm.get_score(lm.tokenize(text))
    pos, neg = s["Positive"], s["Negative"]
    return np.nan if pos + neg < 5 else (pos - neg) / (pos + neg)


# ---------- build the features ----------
tr = pd.read_csv("data/filtered_transcripts_clean.csv")
tr["doc_id"] = tr.index
rows = []
for doc_id, text in zip(tr["doc_id"], tr["transcript"]):
    secs = split_sections(str(text))
    if secs["qa"] is None:
        rows.append(dict(doc_id=doc_id, lm_prep=np.nan, lm_qa=np.nan, lm_qa1=np.nan, lm_qa2=np.nan))
        continue
    sents = sentences(secs["qa"])
    half = len(sents) // 2
    rows.append(dict(doc_id=doc_id, lm_prep=tone(secs["prep"]), lm_qa=tone(secs["qa"]),
                     lm_qa1=tone(" ".join(sents[:half])), lm_qa2=tone(" ".join(sents[half:]))))
f = pd.DataFrame(rows)
f["d_qa_prep"] = f["lm_qa"] - f["lm_prep"]
f["d_qa_halves"] = f["lm_qa2"] - f["lm_qa1"]

ev = pd.read_csv("data/events.csv", parse_dates=["t0"])
sc = pd.read_csv("data/sentiment_scores.csv")[["doc_id", "lm_all"]]
d = ev.merge(sc, on="doc_id").merge(f, on="doc_id").sort_values("t0").reset_index(drop=True)
d["year"] = d["t0"].dt.year
FEATS = ["d_qa_prep", "d_qa_halves"]
print(f"Calls: {len(d)} | with d_qa_prep: {d['d_qa_prep'].notna().sum()} | "
      f"with d_qa_halves: {d['d_qa_halves'].notna().sum()} | companies: {d['ticker'].nunique()}\n")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def partial_rho(df, xcol, ycol="abs_ret"):
    df = df.dropna(subset=[xcol, ycol, "pre_vol"])
    cols = [np.ones(len(df)), rankdata(df["pre_vol"])]
    for dcol in ("year", "ticker"):
        dm = pd.get_dummies(df[dcol], drop_first=True).astype(float)
        cols += [dm[c].values for c in dm.columns]
    Z = np.column_stack(cols)
    res = []
    for col in (xcol, ycol):
        r = rankdata(df[col])
        res.append(r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0])
    return float(np.corrcoef(res[0], res[1])[0, 1])


# ---------- A + B ----------
rows = []
for feat in FEATS:
    s = d.dropna(subset=[feat]).reset_index(drop=True)
    for target in ("abs_ret", "signed_ret"):
        x, y = s[feat].values, s[target].values
        obs = spearmanr(x, y)[0]
        idx = np.arange(len(s))
        boots = [spearmanr(x[i], y[i])[0] for i in (RNG.choice(idx, len(idx)) for _ in range(N_BOOT))]
        groups = s.groupby("ticker").indices
        perm = []
        for _ in range(N_PERM):
            xp = x.copy()
            for _, g in groups.items():
                xp[g] = RNG.permutation(xp[g])
            perm.append(abs(spearmanr(xp, y)[0]))
        p = (1 + np.sum(np.array(perm) >= abs(obs))) / (1 + N_PERM)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        row = dict(feature=feat, target=target, n=len(s), spearman_rho=obs, ci_low=lo, ci_high=hi, perm_p=p)
        if target == "abs_ret":
            row["partial_rho_prevol_year_company"] = partial_rho(s, feat)
        rows.append(row)
g = pd.DataFrame(rows)
g["p_holm"] = holm(g["perm_p"])
g.round(4).to_csv("results/G_within_call.csv", index=False)
print("A/B. Within-call tone features (Spearman, 95% bootstrap CI, permutation p, Holm over 4 tests):")
print(g.round(3).to_string(index=False), "\n")


# ---------- C: out-of-sample ----------
def fit(X, y):
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


rows = []
for feat in FEATS:
    s = d.dropna(subset=[feat]).reset_index(drop=True)
    k = int(0.7 * len(s))
    a, b = s.iloc[:k], s.iloc[k:]
    b0 = fit(a[["pre_vol"]].values, a["abs_ret"].values)
    b1 = fit(a[["pre_vol", feat]].values, a["abs_ret"].values)
    e0 = (b["abs_ret"].values - pred(b0, b[["pre_vol"]].values)) ** 2
    e1 = (b["abs_ret"].values - pred(b1, b[["pre_vol", feat]].values)) ** 2
    diff = e0 - e1
    bs = [diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append(dict(feature=feat, n_train=len(a), n_test=len(b), mse_baseline=e0.mean(), mse_with_feature=e1.mean(),
                     mean_improvement=diff.mean(), ci=f"[{lo:.3f}, {hi:.3f}]", H5_supported=bool(lo > 0)))
o = pd.DataFrame(rows)
o.round(4).to_csv("results/G_within_call_oos.csv", index=False)
print("C. Out-of-sample (time-ordered 70/30). H5 supported only if the CI is entirely above 0:")
print(o.round(3).to_string(index=False), "\n")
print("Report ALL of this, whichever way it goes. Decision rules are in HYPOTHESES_v2.md.")
