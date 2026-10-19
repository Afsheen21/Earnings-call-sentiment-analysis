"""
STEP 4 (prior-work upgrade) - Is your volatility forecaster's R2 = 0.67 real? Add baselines.

Run from your project root (needs internet for yfinance, or pass --csv with Date,Close columns):
    python study/04_volatility_baselines.py                  # downloads NVDA 2015-2025
    python study/04_volatility_baselines.py --ticker NVDA --horizon 5
    python study/04_volatility_baselines.py --csv data/nvda.csv

WHY THIS EXISTS
A professor reading "Random Forest, R2 = 0.67" asks three questions:
  1. 0.67 compared with WHAT? (a model with no baseline proves nothing)
  2. Was the test set in the FUTURE of the training set? (a random split leaks information,
     because volatility is persistent and neighbouring days look alike)
  3. Did the target windows overlap between train and test?
This script answers all three with the same data.

TARGET   : realised volatility over the NEXT `horizon` trading days (std of daily log returns,
           annualised x sqrt(252)). Known only after the window closes.
FEATURES : only information available at day t (vol over past 1, 5, 22 days; past returns).
MODELS   : persistence (naive: "tomorrow's vol = last week's vol"), HAR (simple regression,
           the standard finance benchmark), Random Forest.
SPLITS   : (a) time-ordered 70/30 with a purge gap so no window overlaps  <- the honest number
           (b) random shuffle split                                       <- shows how leakage inflates R2
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ap = argparse.ArgumentParser()
ap.add_argument("--ticker", default="NVDA")
ap.add_argument("--csv", default=None)
ap.add_argument("--start", default="2015-01-01")
ap.add_argument("--end", default="2025-12-31")
ap.add_argument("--horizon", type=int, default=5)
args = ap.parse_args()
H, SEED = args.horizon, 42

# ---------- 1. load prices ----------
if args.csv:
    px = pd.read_csv(args.csv, parse_dates=["Date"]).set_index("Date")["Close"]
else:
    import yfinance as yf
    df = yf.download(args.ticker, start=args.start, end=args.end, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]   # flatten MultiIndex
    px = df["Close"]
px = px.dropna().astype(float)
r = np.log(px).diff()

# ---------- 2. features (past only) and target (future only) ----------
d = pd.DataFrame(index=px.index)
d["rv1"] = r.abs() * np.sqrt(252)                       # yesterday-sized shock
d["rv5"] = r.rolling(5).std() * np.sqrt(252)            # past week
d["rv22"] = r.rolling(22).std() * np.sqrt(252)          # past month
d["ret1"] = r
d["ret5"] = r.rolling(5).sum()
# target: std of the NEXT H returns, i.e. returns at t+1 ... t+H
d["y"] = r[::-1].rolling(H).std()[::-1].shift(-1) * np.sqrt(252)
d = d.dropna()
FEATS_HAR = ["rv1", "rv5", "rv22"]
FEATS_RF = ["rv1", "rv5", "rv22", "ret1", "ret5"]
print(f"{len(d)} usable days, {d.index[0].date()} to {d.index[-1].date()}, horizon {H} days\n")


def qlike(y, f):
    """QLIKE loss on variances: standard for volatility forecasts (lower is better)."""
    y2, f2 = np.maximum(y ** 2, 1e-8), np.maximum(f ** 2, 1e-8)
    return float(np.mean(np.log(f2) + y2 / f2))


def score(y, f):
    return dict(R2=r2_score(y, f), RMSE=float(np.sqrt(mean_squared_error(y, f))), QLIKE=qlike(y, f))


def run(train, test, label):
    rows = []
    rows.append(dict(split=label, model="persistence (rv5)", **score(test["y"], test["rv5"])))
    har = LinearRegression().fit(train[FEATS_HAR], train["y"])
    rows.append(dict(split=label, model="HAR regression", **score(test["y"], har.predict(test[FEATS_HAR]))))
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rf.fit(train[FEATS_RF], train["y"])
    rows.append(dict(split=label, model="Random Forest", **score(test["y"], rf.predict(test[FEATS_RF]))))
    return rows


rows = []
# (a) honest: time-ordered split with a purge gap of H days so no target window overlaps
cut = int(len(d) * 0.70)
train_t, test_t = d.iloc[: cut - H], d.iloc[cut:]
rows += run(train_t, test_t, "time-ordered 70/30 (honest)")

# (b) leaky: random shuffle split (what an untested pipeline often does)
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(d))
cut_r = int(len(d) * 0.70)
rows += run(d.iloc[idx[:cut_r]], d.iloc[idx[cut_r:]], "random shuffle (leaky)")

res = pd.DataFrame(rows)
pd.set_option("display.width", 140)
print(res.round(4).to_string(index=False))
import os
os.makedirs("results", exist_ok=True)
res.to_csv("results/volatility_baselines.csv", index=False)

honest = res[res.split.str.startswith("time")].set_index("model")
print("\nHOW TO READ THIS")
print(f"- Honest RF R2 = {honest.loc['Random Forest','R2']:.3f}. Compare it with persistence "
      f"({honest.loc['persistence (rv5)','R2']:.3f}) and HAR ({honest.loc['HAR regression','R2']:.3f}).")
print("- If RF barely beats HAR, say so: 'RF adds X over the HAR benchmark'. That is a real, defensible claim.")
print("- If the leaky R2 is much higher than the honest one, your old 0.67 was probably inflated; report the honest number.")
print("- Put the honest table in your README and in your research statement.")
