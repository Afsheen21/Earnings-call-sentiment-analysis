import pandas as pd

print("Step 1: starting")

df = pd.read_csv("data/combined_dataset.csv")
print("Step 2: loaded combined_dataset.csv, shape:", df.shape)

row = df.iloc[0]
ticker = row['ticker']
date = row['clean_date']
print("Step 3: testing ticker", ticker, "date", date)

prices = pd.read_csv(f"data/raw_prices/{ticker}_prices.csv")
print("Step 4: loaded price file, shape:", prices.shape)

prices['Date'] = pd.to_datetime(prices['Date'], utc=True).dt.tz_localize(None)
print("Step 5: converted dates successfully")

print("DONE")


import pandas as pd

print("Starting...")

df = pd.read_csv("data/combined_dataset.csv")

def get_price_change(ticker, earnings_date, days_after):
    try:
        prices = pd.read_csv(f"data/raw_prices/{ticker}_prices.csv")
        prices['Date'] = pd.to_datetime(prices['Date'], utc=True).dt.tz_localize(None)
    except FileNotFoundError:
        return None
    
    earnings_date = pd.to_datetime(earnings_date)
    
    window = prices[
        (prices['Date'] >= earnings_date - pd.Timedelta(days=2)) &
        (prices['Date'] <= earnings_date + pd.Timedelta(days=days_after + 2))
    ]
    
    if window.empty or len(window) < 2:
        return None
    
    price_before = window.iloc[0]['Close']
    price_after = window.iloc[-1]['Close']
    return round(((price_after - price_before) / price_before) * 100, 2)

changes_1d = []

for i, row in df.iterrows():
    c1 = get_price_change(row['ticker'], row['clean_date'], 1)
    changes_1d.append(c1)
    print(f"Processed {i+1}/{len(df)}")

print("LOOP FINISHED")

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("data/combined_dataset.csv")

def get_price_change(ticker, earnings_date, days_after):
    try:
        prices = pd.read_csv(f"data/raw_prices/{ticker}_prices.csv")
        prices['Date'] = pd.to_datetime(prices['Date'], utc=True).dt.tz_localize(None)
    except FileNotFoundError:
        return None
    
    earnings_date = pd.to_datetime(earnings_date)
    
    window = prices[
        (prices['Date'] >= earnings_date - pd.Timedelta(days=2)) &
        (prices['Date'] <= earnings_date + pd.Timedelta(days=days_after + 2))
    ]
    
    if window.empty or len(window) < 2:
        return None
    
    price_before = window.iloc[0]['Close']
    price_after = window.iloc[-1]['Close']
    return round(((price_after - price_before) / price_before) * 100, 2)

changes_1d = []
changes_30d = []

for i, row in df.iterrows():
    changes_1d.append(get_price_change(row['ticker'], row['clean_date'], 1))
    changes_30d.append(get_price_change(row['ticker'], row['clean_date'], 30))
    print(f"Processed {i+1}/{len(df)}")

df['price_change_1d'] = changes_1d
df['price_change_30d'] = changes_30d

print("1-day correlation:", df['sentiment'].corr(df['price_change_1d']))
print("7-day correlation:", df['sentiment'].corr(df['price_change_7d']))
print("30-day correlation:", df['sentiment'].corr(df['price_change_30d']))

df.to_csv("data/combined_dataset.csv", index=False)
print("Saved with multiple time windows!")

plt.figure(figsize=(8,6))
plt.scatter(df['sentiment'], df['price_change_7d'], alpha=0.6)
plt.xlabel('Sentiment Score')
plt.ylabel('Price Change % (7 days after)')
plt.title('Earnings Call Sentiment vs Stock Price Change')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.savefig('data/sentiment_vs_price.png')
plt.close()
print("Chart saved as data/sentiment_vs_price.png")

import pandas as pd

df = pd.read_csv("data/combined_dataset.csv")

for ticker in df['ticker'].unique():
    subset = df[df['ticker'] == ticker]
    if len(subset) >= 5:  # only check companies with enough data points
        corr = subset['sentiment'].corr(subset['price_change_7d'])
        print(f"{ticker}: correlation = {corr:.3f} (n={len(subset)})")