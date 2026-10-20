# Hypotheses, round 2 (written BEFORE running 06_within_call.py)

Date written: ____ (commit and push this file BEFORE you run step 6; the commit time is the proof)

**Why this round exists.** Round 1 (HYPOTHESES.md, commit 4b64641) found that Loughran-McDonald (LM) net tone is
associated with the size of the price move (rho = 0.38) but does not improve out-of-sample prediction over
pre-event volatility. My leading explanation is that one averaged score hides the signal, and that the information
sits in how tone CHANGES inside a call. This round tests that explanation directly. Only LM is used, because it is the
only scorer that survived controls in round 1. I will not add other scorers after seeing results.

**Features (all LM net tone = (pos - neg) / (pos + neg)).**
- `d_qa_prep` = tone of the Q&A section minus tone of the prepared remarks.
- `d_qa_halves` = tone of the second half of the Q&A (by sentence count) minus tone of the first half.

**Targets.** `abs_ret` (size of the move) and `signed_ret` (direction), defined exactly as in round 1.

**H4.** Each feature has a non-zero Spearman association with `abs_ret` or `signed_ret` in the full sample.
Four tests (2 features x 2 targets), Holm-corrected, 5% level, within-company permutation p-values.

**H5.** Adding a feature to `pre_vol` improves out-of-sample prediction of `abs_ret` (time-ordered 70/30 split,
paired bootstrap CI of the error difference entirely above 0).

**Decision rules (fixed in advance).**
- "Supported" requires the Holm-corrected p < 0.05 (H4) or the CI above 0 (H5). Anything else is reported as not supported.
- If H5 fails, I report a null result and do not add features, change the split or switch scorers until something works.
- Every result is reported, in the same tables, whichever way it goes.

**Known limits (stated up front).** About 139 calls with a usable Q&A split from 10 companies; one scorer;
two features; one baseline; one event definition. A positive result here would be a reason to test on a larger
sample, not a finding to apply.
