"""
STEP 2 - Score every transcript with three sentiment methods.

  vader   : general-purpose lexicon (what your project already used)
  lm      : Loughran-McDonald finance dictionary (words classified for finance)
  finbert : FinBERT, a language model trained on financial text (ProsusAI/finbert)

Each method is computed for three parts of a call:  all / prep (prepared remarks) / qa (Q&A).
Results are saved to data/sentiment_scores.csv, one row per transcript (doc_id).

Run from the ROOT of your project folder:
    python study/02_score_sentiment.py                    # all three methods
    python study/02_score_sentiment.py --methods vader,lm # skip FinBERT (fast test)
    python study/02_score_sentiment.py --limit 3          # smoke test on 3 transcripts first
FinBERT is slow on a CPU (30-60+ minutes). Run it on free Google Colab if you can.
"""
import argparse
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

TRANSCRIPTS = "data/filtered_transcripts_clean.csv"
OUT = "data/sentiment_scores.csv"
QA_MARK = re.compile(r"question[\s\-–]*and[\s\-–]*answer|questions?\s*(?:&|and)\s*answers?|\bQ\s*&\s*A\b", re.I)


def split_sections(text):
    """Return {'all','prep','qa'} text. qa is None if no Q&A marker is found."""
    m = QA_MARK.search(text, int(len(text) * 0.15))
    if m is None:
        return {"all": text, "prep": None, "qa": None}
    return {"all": text, "prep": text[:m.start()], "qa": text[m.start():]}


def sentences(text):
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [s.strip() for s in parts if len(s.split()) >= 4]


def add_sentence_method(res, secs, scorer, name):
    """Score each sentence once, then average per part (prep / qa / all)."""
    cols = {"all": [], "prep": [], "qa": []}
    for s in tqdm(secs, desc=name):
        if s["qa"] is None:                       # no Q&A marker found: only "all" is available
            sents = sentences(s["all"])
            sc = scorer(sents) if sents else []
            cols["all"].append(np.mean(sc) if sc else np.nan)
            cols["prep"].append(np.nan)
            cols["qa"].append(np.nan)
            continue
        sp, sq = sentences(s["prep"]), sentences(s["qa"])
        sc = scorer(sp + sq) if (sp or sq) else []
        scp, scq = sc[:len(sp)], sc[len(sp):]
        cols["all"].append(np.mean(sc) if sc else np.nan)
        cols["prep"].append(np.mean(scp) if scp else np.nan)
        cols["qa"].append(np.mean(scq) if scq else np.nan)
    for part in cols:
        res[f"{name}_{part}"] = cols[part]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="vader,lm,finbert")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N transcripts (smoke test)")
    args = ap.parse_args()
    methods = args.methods.split(",")

    df = pd.read_csv(TRANSCRIPTS)
    if args.limit:
        df = df.head(args.limit)
    df["doc_id"] = df.index
    secs = [split_sections(str(t)) for t in df["transcript"]]
    res = pd.DataFrame({"doc_id": df["doc_id"]})

    if "vader" in methods:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        vader = lambda sents: [sia.polarity_scores(x)["compound"] for x in sents]
        add_sentence_method(res, secs, vader, "vader")

    if "lm" in methods:
        import pysentiment2 as ps
        lm = ps.LM()
        for part in ["all", "prep", "qa"]:
            vals = []
            for s in tqdm(secs, desc=f"LM {part}"):
                if not s[part]:
                    vals.append(np.nan)
                    continue
                sc = lm.get_score(lm.tokenize(s[part]))
                vals.append((sc["Positive"] - sc["Negative"]) / (sc["Positive"] + sc["Negative"] + 1e-6))
            res[f"lm_{part}"] = vals

    if "finbert" in methods:
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        print("FinBERT device:", "GPU" if device == 0 else "CPU (slow)")
        clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None,
                       truncation=True, max_length=256, batch_size=32, device=device)

        def finbert(sents):
            out = clf(sents)
            return [next(d["score"] for d in o if d["label"] == "positive") -
                    next(d["score"] for d in o if d["label"] == "negative") for o in out]
        add_sentence_method(res, secs, finbert, "finbert")

    res.to_csv(OUT, index=False)
    print(f"Saved {OUT}")
    print(res.describe().round(3).T)


if __name__ == "__main__":
    main()
