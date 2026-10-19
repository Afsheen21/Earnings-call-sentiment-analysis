"""
STEP 1 - Build the "event table": what the stock did around each earnings call.

For each call we compute three numbers (all from daily closing prices):
  signed_ret : log(Close[t0+1] / Close[t0-1])   -> direction + size of the move
  abs_ret    : |signed_ret|                      -> size of the move only (our main target)
  pre_vol    : std. dev. of the 20 daily returns BEFORE the event window (baseline volatility)

t0 = first trading day on or after the call date. Using t0-1 -> t0+1 covers the reaction
whether the call was before the market opened or after it closed.
pre_vol only uses returns up to day t0-1, so it never overlaps the event window (no leakage).

Run from the ROOT of your Earnings-call-sentiment-analysis folder:
    python study/01_prepare_events.py
"""
import os
import numpy as np
import pandas as pd

TRANSCRIPTS = "data/filtered_transcripts_clean.csv"
PRICE_DIR = "data/raw_prices"
OUT = "data/events.csv"


def load_prices(ticker, start, end):
    """Use your saved price file if it exists; otherwise download it with yfinance."""
    path = f"{PRICE_DIR}/{ticker}_prices.csv"
    if os.path.exists(path):
        p = pd.read_csv(path)
    else:
        import yfinance as yf
        os.makedirs(PRICE_DIR, exist_ok=True)
        p = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(p.columns, pd.MultiIndex):      # newer yfinance versions
            p.columns = p.columns.get_level_values(0)
        p = p.reset_index()
        p.to_csv(path, index=False)
    p["Date"] = pd.to_datetime(p["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    p = p.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    p["logret"] = np.log(p["Close"]).diff()
    return p


def main():
    df = pd.read_csv(TRANSCRIPTS)
    df["doc_id"] = df.index                      # the ID that links every later file
    df["call_date"] = pd.to_datetime(df["clean_date"])
    rows, skipped = [], 0
    for ticker, g in df.groupby("ticker"):
        start = (g["call_date"].min() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        end = (g["call_date"].max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        try:
            p = load_prices(ticker, start, end)
        except Exception as e:
            print(f"  could not get prices for {ticker}: {e}")
            skipped += len(g)
            continue
        for _, r in g.iterrows():
            i = int(p["Date"].searchsorted(r["call_date"]))   # first trading day >= call date
            if i < 22 or i + 1 >= len(p) or (p["Date"].iloc[i] - r["call_date"]).days > 5:
                skipped += 1
                continue
            signed = float(np.log(p["Close"].iloc[i + 1] / p["Close"].iloc[i - 1]))
            pre_vol = float(p["logret"].iloc[i - 20:i].std())
            rows.append(dict(doc_id=r["doc_id"], ticker=ticker, t0=p["Date"].iloc[i],
                             signed_ret=signed * 100, abs_ret=abs(signed) * 100,
                             pre_vol=pre_vol * 100))
    out = pd.DataFrame(rows).sort_values("doc_id")
    out.to_csv(OUT, index=False)
    print(f"Saved {OUT}: {len(out)} calls kept, {skipped} skipped (no prices / too close to data edge)")
    print(out[["signed_ret", "abs_ret", "pre_vol"]].describe().round(3))


if __name__ == "__main__":
    main()
