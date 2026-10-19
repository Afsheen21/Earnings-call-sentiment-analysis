# Plan: improve your research and prior work (Watanabe's feedback)

Watanabe said: (1) prior work lacks substance, (2) related work is thin. This plan fixes both.

## Day 1 (today) - Study setup, 1-2 hours
1. Copy this `study` folder into your project root. `pip install pandas numpy scipy scikit-learn matplotlib tqdm vaderSentiment pysentiment2 yfinance transformers torch`
2. Edit HYPOTHESES.md; commit and push it ("Pre-registered hypotheses").
3. `python study/01_prepare_events.py` (downloads prices).
4. `python study/02_score_sentiment.py --limit 3 --methods vader,lm` (smoke test), then without --limit.

## Day 2 - Finish the study
5. FinBERT on Colab: `python study/02_score_sentiment.py --methods vader,lm,finbert`; download data/sentiment_scores.csv.
6. `python study/03_analysis.py`. Read results honestly (CI crossing 0 = no evidence).

## Day 3 - Prior work evidence (the volatility project)
7. `python study/04_volatility_baselines.py` (NVDA). Put the honest time-ordered table in the volatility-forecaster README. If the leaky R2 is far above the honest R2, your old 0.67 was inflated: report the honest one and say what you changed.

## Day 4 - Write up
8. README for the sentiment study: question, data, method, results table, figure, limitations.
9. Read RELATED_WORK_DRAFT.md, open each paper, fix details, keep only claims you checked.
10. Send me: results tables + the honest volatility table. I write research statement v4 (2 pages) and the final emails.

## Honest limits
- Step 2 was only syntax-checked; steps 1, 3 and 4 were tested on synthetic data. Report any error with the full message.
- 148 calls / 10 companies is small. A null result is a valid finding; do not change hypotheses after seeing results.
