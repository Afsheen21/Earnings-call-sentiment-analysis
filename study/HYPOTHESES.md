# Hypotheses (written BEFORE running the analysis)

Date written: 15-oct-2026 (commit this file to GitHub before you run 03_analysis.py - the commit time is your proof)

**Question.** Do finance-specific sentiment measures capture something about earnings calls that a
general-purpose lexicon (VADER) misses?

**Target.** Size of the stock move around the call (`abs_ret`, log return from t0-1 to t0+1, in %),
and also its direction (`signed_ret`). Baseline control: volatility over the 20 days before the call (`pre_vol`).

**H1.** Finance-specific scorers (Loughran-McDonald, FinBERT) have a stronger association with `abs_ret`
than VADER does (|rho| larger, bootstrap CI of the difference above 0).

**H2.** Adding a sentiment score to `pre_vol` improves out-of-sample prediction of `abs_ret`
(time-ordered split, 70% train / 30% test).

**H3.** Scores from the Q&A section are more strongly associated with `abs_ret` than scores from the prepared remarks.

**Decision rules (fixed in advance).** Tests use a 5% level; the six main tests (3 methods x 2 targets) are
Holm-corrected. If no result is significant, I will report that as a null result and will not change the
method until something works.

**Known limits (stated up front).** 148 calls from 10 companies; only one baseline model; one event definition.
