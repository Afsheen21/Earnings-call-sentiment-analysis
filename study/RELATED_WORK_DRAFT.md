# Related work (draft v1) - every claim below was read on the page linked in the references

**Tone of earnings calls and returns.** Price et al. (2012) showed that the textual tone of earnings calls predicts abnormal returns and trading volume, that the Q&A adds explanatory power for post-announcement drift, and that a context-specific dictionary beat the general Harvard IV-4 list. Matsumoto, Pronk and Roelofsen (2011), using more than 10,000 transcripts, found both the managers' presentation and the analysts' discussion informative, with the discussion relatively more so. Davis et al. (2015) found that call tone has a manager-specific component beyond firm performance and incentives, and Druz, Wagner and Zeckhauser (2015) found that prices respond to negative word choice and that a "tone surprise" predicts future earnings. Kim and Park (preliminary version of a 2026 paper) link FinBERT-scored Q&A tone to cumulative abnormal returns for 36 Korean firms (1,232 calls) that hold English calls.

**Scoring methods.** Loughran and McDonald (2011) showed that about three-quarters of the words a general dictionary labels negative are not negative in a finance context, and built finance-specific word lists. FinBERT (Araci, 2019) improved on prior results for financial sentiment classification, and Huang, Wang and Yang (2023) report that it beats the Loughran-McDonald dictionary on analyst-report sentences. VADER (Hutto and Gilbert, 2014) was built and validated on social-media text, so applying it to long call transcripts is a domain transfer.

**Text and volatility.** Kogan et al. (2009) predicted volatility from 26,806 10-K sections and reported that text alone works nearly as well as a past-volatility baseline and often better combined with it (from the authors' slides). Qin and Yang (2019) combined transcript and audio cues from S&P 500 calls and reduced volatility prediction error compared with text alone. Jang and Jeong (KAFE 2023 version) add a news-embedding correction to a HAR baseline and raise out-of-sample R2 from 0.4186 to 0.5710, while FinBERT sentiment alone raises it only to 0.4290. Recent LLM-era work includes ECC Analyzer (Cao et al., 2024), which combines an LLM with audio features, and Nakatsuka and Suimon (2025), who relate ChatGPT-processed Q&A sentiment to abnormal returns in Japanese briefings.

**Gap.** In the papers reviewed here, evidence on call tone is mostly in-sample association with abnormal returns, and the papers that test out-of-sample prediction use either 10-Ks or news, or require audio and large models. None of them compares a general-purpose scorer (VADER), a finance dictionary (Loughran-McDonald) and a finance language model (FinBERT) on the same calls against a pre-event volatility baseline with a time-ordered split, while also separating prepared remarks from Q&A. This study does that on a small sample, and treats the outcome, including a null result, as informative about whether sentiment adds anything beyond volatility itself.

## References (URL read)
1. Price, Doran, Peterson, Bliss (2012), J. Banking & Finance 36(4) - https://ideas.repec.org/a/eee/jbfina/v36y2012i4p992-1011.html
2. Matsumoto, Pronk, Roelofsen (2011), The Accounting Review 86(4) - https://pure.eur.nl/en/publications/what-makes-conference-calls-useful-the-information-content-of-man/
3. Davis, Ge, Matsumoto, Zhang (2015), Review of Accounting Studies 20(2) - https://ideas.repec.org/a/spr/reaccs/v20y2015i2d10.1007_s11142-014-9309-4.html
4. Druz, Wagner, Zeckhauser (2015), NBER WP 20991 - https://www.nber.org/papers/w20991.pdf
5. Loughran, McDonald (2011), J. Finance 66(1) - https://papers.ssrn.com/abstract=1331573
6. Araci (2019), FinBERT, arXiv:1908.10063 - https://arxiv.org/abs/1908.10063
7. Huang, Wang, Yang (2023), Contemporary Accounting Research - read via a mirror: https://academicnewsletter.sufe.edu.cn/info/410757 (open the publisher page yourself before citing)
8. Hutto, Gilbert (2014), ICWSM 8(1) - https://ojs.aaai.org/index.php/icwsm/article/view/14550
9. Kogan, Levin, Routledge, Sagi, Smith (2009), NAACL-HLT - slides: https://homes.cs.washington.edu/~nasmith/slides/kogan+levin+routledge+sagi+smith.naacl09.pdf
10. Qin, Yang (2019), ACL - https://preview.aclanthology.org/menus/P19-1038/
11. Jang, Jeong, KAFE 2023 version - https://www.kafe2023.org/bbs/reference/260/download/527
12. Kim, Park, preliminary version of 2026 paper - https://iksa.or.kr/file/download/7220
13. Cao et al. (2024), ECC Analyzer, arXiv:2404.18470 - https://arxiv.org/abs/2404.18470
14. Nakatsuka, Suimon (2025), Digital Finance 7(4) - https://ideas.repec.org/a/spr/digfin/v7y2025i4d10.1007_s42521-025-00159-y.html

## Before you use it
- Open the papers marked "mirror" or "slides" on the publisher page, and fix any year or venue that differs.
- "Gap" is a claim about the papers I could read. Do not write "no study has ever..." in the statement.
- Frankel, Johnson and Skinner (1999) was dropped because no abstract could be read.
