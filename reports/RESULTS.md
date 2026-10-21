# My experiment results

I kept the full numbers here so I can compare the models without relying on memory. These tables come from the saved JSON reports; the source CSVs are unchanged.

## Dataset audit

| Task | Raw training rows | Training | Validation | Test | Trainable labels |
|---|---:|---:|---:|---:|---:|
| call_scam | 10000 | 445 | 149 | 149 | 4 |
| complaint | 93686 | 62126 | 15532 | 30771 | 39 |

Complaint test uses the supplied `test.csv`. Overlapping normalized templates are removed from training, not from test.

Complaint training rows removed for official-test overlap: **12778**.
Complaint test repeated templates: **3651**; conflicting templates: **130**.

Five contradictory Hinglish subtypes are merged into `other_scam`; see `call_label_conflicts.csv`.
Normalized duplicates are controlled, but semantic paraphrases may still cross splits.

## Held-out metrics

All scores below are fractions, not percentages. Typo tests use synthetic internal-letter swaps in 25% of eligible words.

| Task | Model | Validation macro-F1 | Test macro-F1 | Typo test macro-F1 | Test accuracy | Seconds |
|---|---|---:|---:|---:|---:|---:|
| call_scam | TF-IDF word (selected baseline) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.4 |
| call_scam | TF-IDF char | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5 |
| call_scam | TF-IDF hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6 |
| call_scam | word2vec | 1.0000 | 1.0000 | 0.9935 | 1.0000 | 12.7 |
| call_scam | subword | 1.0000 | 1.0000 | 0.9935 | 1.0000 | 6.6 |
| call_scam | multilingual DistilBERT | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 39.7 |
| complaint | TF-IDF word | 0.2488 | 0.2259 | 0.2186 | 0.4952 | 114.8 |
| complaint | TF-IDF char | 0.2512 | 0.2193 | 0.2189 | 0.4793 | 376.2 |
| complaint | TF-IDF hybrid (selected baseline) | 0.2631 | 0.2345 | 0.2269 | 0.5014 | 399.8 |
| complaint | word2vec | 0.1290 | 0.1205 | 0.1115 | 0.2918 | 116.3 |
| complaint | subword | 0.1328 | 0.1210 | 0.1095 | 0.3002 | 203.2 |
| complaint | multilingual DistilBERT | 0.1677 | 0.1609 | 0.1578 | 0.4883 | 973.9 |
| complaint | financial expert + baseline | 0.2639 | 0.2311 | 0.2253 | 0.5028 | 187.6 |

## Harder call evaluation: template families

I grouped the 743 messages into **77** near-duplicate families before evaluating the frozen baseline with five-fold group cross-validation.
Family-held-out macro-F1: **0.6308**; typo macro-F1: **0.6337**; accuracy: **0.7416**.

Each message is evaluated exactly once; no family crosses a fold's training and test sets. All four classes are represented in training for each recorded fold.
The 1.0000 call scores above came from the earlier exact-deduplicated split. They were too optimistic because related templates remained on both sides. I keep them for comparison, but use the family-held-out result in the README headline.
Family grouping uses a 0.80 character-TF-IDF cosine threshold and connected components. This is a heuristic stress test, not a human-verified semantic split or an independent real-world benchmark.

Default inference uses validation-selected backends:

- call_scam: **baseline** (selection score 1.0000).
- complaint: **baseline** (selection score 0.2579).

Additional word-feature optimizer checks (validation only):

- sgd_balanced_1e4: macro-F1 0.2163, accuracy 0.4174.
- sgd_unweighted_1e5: macro-F1 0.1887, accuracy 0.5067.
- linear_svc_balanced: macro-F1 0.2238, accuracy 0.4531.

Baseline selection uses the mean of clean and noisy validation macro-F1, never test scores.
Word2Vec/subword use fixed learning-experiment settings, not an exhaustive hyperparameter search.
A perfect result on the small templated call dataset does not establish real-world or cross-language reliability.

## Selected baseline: weakest complaint classes

| Class | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| Crime Against Women & Children :: Computer Generated CSAM/CSEM | 0.000 | 0.000 | 0.000 | 2 |
| Crime Against Women & Children :: Cyber Blackmailing & Threatening | 0.000 | 0.000 | 0.000 | 1 |
| Crime Against Women & Children :: Sexual Harassment | 0.000 | 0.000 | 0.000 | 1 |
| Hacking  Damage to computercomputer system etc :: Tampering with computer source documents | 0.000 | 0.000 | 0.000 | 14 |
| Online Cyber Trafficking :: Online Trafficking | 0.000 | 0.000 | 0.000 | 61 |
| Online and Social Media Related Crime :: Intimidating Email | 0.000 | 0.000 | 0.000 | 11 |
| RapeGang Rape RGRSexually Abusive Content :: Unspecified | 0.163 | 0.018 | 0.032 | 911 |
| Hacking  Damage to computercomputer system etc :: Damage to computer computer systems etc | 0.056 | 0.029 | 0.038 | 35 |
| Cyber Terrorism :: Cyber Terrorism | 0.091 | 0.039 | 0.055 | 51 |
| Sexually Explicit Act :: Unspecified | 0.051 | 0.120 | 0.072 | 527 |

## Duplicate-aware official-test sensitivity

- baseline: deduplicated (template, label) test macro-F1 **0.2384** on 27296 examples. Contradictory labels are retained.
- word2vec: deduplicated (template, label) test macro-F1 **0.1222** on 27296 examples. Contradictory labels are retained.
- subword: deduplicated (template, label) test macro-F1 **0.1227** on 27296 examples. Contradictory labels are retained.
- transformer: deduplicated (template, label) test macro-F1 **0.1614** on 27296 examples. Contradictory labels are retained.
- hierarchical: deduplicated (template, label) test macro-F1 **0.2350** on 27296 examples. Contradictory labels are retained.

## Small manual diagnostic

Hand-authored mixed-language development examples, not a representative benchmark and never used for fitting or model selection. Expected labels were clarified after inspecting source semantics; collect a fresh independent gold set for a final claim.

- baseline: 11/18 correct. Inspect every example in `manual_challenge.json`, including warning/negation failures.
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.669): This is the police. You are under digital arrest. Transfer money now or we will arrest you.
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.859): mai police se bol rha hu tum digital arrest mein ho turant paise transfer kro
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.584): CBI officer bol raha hun. Video call pe raho, digital arrest se bachne ko payment karo.
  - Expected `police_digital_arrest`, predicted `other_scam` (score 0.464): apka beta police custody me hai paise bhejo warna usko jail bhej denge
  - Expected `police_digital_arrest`, predicted `none` (score 0.627): tumhare bete ko drugs case me pakda hai. Abhi paise do toh chhod denge
  - Expected `none`, predicted `other_scam` (score 0.492): Never share your OTP with anyone. This is a bank security awareness message.
- word2vec: 10/18 correct. Inspect every example in `manual_challenge.json`, including warning/negation failures.
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.994): This is the police. You are under digital arrest. Transfer money now or we will arrest you.
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.966): mai police se bol rha hu tum digital arrest mein ho turant paise transfer kro
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.899): CBI officer bol raha hun. Video call pe raho, digital arrest se bachne ko payment karo.
  - Expected `police_digital_arrest`, predicted `other_scam` (score 0.980): apka beta police custody me hai paise bhejo warna usko jail bhej denge
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.353): tumhare bete ko drugs case me pakda hai. Abhi paise do toh chhod denge
  - Expected `other_scam`, predicted `police_blackmail` (score 0.621): amazon prize jeet gaye ho delivery fees pehle pay karo
- subword: 8/18 correct. Inspect every example in `manual_challenge.json`, including warning/negation failures.
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.955): This is the police. You are under digital arrest. Transfer money now or we will arrest you.
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.915): mai police se bol rha hu tum digital arrest mein ho turant paise transfer kro
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.910): CBI officer bol raha hun. Video call pe raho, digital arrest se bachne ko payment karo.
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.425): apka beta police custody me hai paise bhejo warna usko jail bhej denge
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.691): tumhare bete ko drugs case me pakda hai. Abhi paise do toh chhod denge
  - Expected `other_scam`, predicted `police_blackmail` (score 0.728): amazon prize jeet gaye ho delivery fees pehle pay karo
- transformer: 5/18 correct. Inspect every example in `manual_challenge.json`, including warning/negation failures.
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.324): This is the police. You are under digital arrest. Transfer money now or we will arrest you.
  - Expected `police_digital_arrest`, predicted `none` (score 0.402): mai police se bol rha hu tum digital arrest mein ho turant paise transfer kro
  - Expected `police_digital_arrest`, predicted `none` (score 0.505): apka beta police custody me hai paise bhejo warna usko jail bhej denge
  - Expected `police_digital_arrest`, predicted `none` (score 0.544): tumhare bete ko drugs case me pakda hai. Abhi paise do toh chhod denge
  - Expected `police_blackmail`, predicted `police_digital_arrest` (score 0.314): Your phone number has been used to share obscene videos. Pay us to avoid a police case.
  - Expected `police_blackmail`, predicted `none` (score 0.315): aapke number se gande videos share hue hain turant paise bhejo warna case hoga
- best: 11/18 correct. Inspect every example in `manual_challenge.json`, including warning/negation failures.
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.669): This is the police. You are under digital arrest. Transfer money now or we will arrest you.
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.859): mai police se bol rha hu tum digital arrest mein ho turant paise transfer kro
  - Expected `police_digital_arrest`, predicted `police_blackmail` (score 0.584): CBI officer bol raha hun. Video call pe raho, digital arrest se bachne ko payment karo.
  - Expected `police_digital_arrest`, predicted `other_scam` (score 0.464): apka beta police custody me hai paise bhejo warna usko jail bhej denge
  - Expected `police_digital_arrest`, predicted `none` (score 0.627): tumhare bete ko drugs case me pakda hai. Abhi paise do toh chhod denge
  - Expected `none`, predicted `other_scam` (score 0.492): Never share your OTP with anyone. This is a bank security awareness message.

## Integrity and environment

Verified **19** copied source files against their originals using SHA-256.
See `source_integrity.json` and `environment.json` for exact values.

## Limits

- English and Hinglish label vocabularies differ; cross-language transfer is not independently validated.
- Complaint classification has no benign class; call `none` does not prove safety.
- Rule-based PII masking is incomplete; raw copies/reference notebooks remain sensitive.
- Confidence scores and review threshold are uncalibrated; unknown scams can receive high scores.
- Final reports supersede internal-test exploratory artifacts under `superseded/`.
- No RAG layer, live-call integration, or automatic complaint submission is included.
