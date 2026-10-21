# Evaluation Results

Metrics are generated from the saved experiment reports. Evaluation protocols are listed explicitly because the call and complaint datasets use different taxonomies and split strategies.

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

The 743 messages form **77** near-duplicate families. The fixed baseline is evaluated with five-fold group cross-validation.
Family-held-out macro-F1: **0.6308**; typo macro-F1: **0.6337**; accuracy: **0.7416**.

Each message is evaluated exactly once; no family crosses a fold's training and test sets. All four classes are represented in training for each recorded fold.
The 1.0000 call scores above use the earlier exact-deduplicated split. Related templates crossed that split, producing optimistic estimates. The primary call result uses family-held-out evaluation.
Family grouping uses a 0.80 character-TF-IDF cosine threshold and connected components. This is a heuristic stress test, not a human-verified semantic split or an independent real-world benchmark.

## Bounded call improvement experiment

Six fixed candidates are compared within three inner family folds. Five outer family folds assess the selection procedure. Vocabularies, scalers and classifiers are fitted only on the corresponding training fold.

| Model procedure | Clean macro-F1 | Typo macro-F1 | Accuracy |
|---|---:|---:|---:|
| Reference word baseline | 0.6308 | 0.6337 | 0.7416 |
| Nested candidate selection | 0.8256 | 0.7709 | 0.8654 |

The approximate paired family-bootstrap 95% interval for the **clean F1 difference** is **[-0.0283, 0.4064]** (500 resamples). This is wide and includes zero.
Promotion criteria: Mean gain >=0.02; paired clean family-bootstrap lower bound >0; typo F1 drop <=0.005. Passed: **False**.
The optional `robust` artifact uses **frozen_c01**, selected with a separate grouped inner search and refitted on all 743 development messages. Its training-set performance is not the reported nested score.
The baseline remains the default. The encoder is frozen; only a small L2-regularized classification head learns labels. This reduces capacity but does not prove absence of overfitting.
This corpus was explored before this experiment, families are heuristic, and several classes have few independent families. Fresh human-labelled English/Hinglish messages are still needed for independent confirmation.

## Bounded complaint improvement experiment

Three fixed heads use the same training-only TF-IDF features. Selection uses mean clean/noisy validation macro-F1, with a minimum gain of 0.02 before promotion.

| Head | Clean validation macro-F1 | Typo validation macro-F1 | Selection mean |
|---|---:|---:|---:|
| complement_nb | 0.1635 | 0.1687 | 0.1661 |
| l2_balanced | 0.2501 | 0.2443 | 0.2472 |
| l2_unweighted | 0.2028 | 0.1896 | 0.1962 |

Reference selection mean: **0.2579**. Selected head: **l2_balanced**. Promotion rule passed: **False**.
Not evaluated: no material validation improvement. Existing baseline retained.
The validation set has been used in earlier development, so these are development results. Labels and difficult classes are retained; test scores do not select settings.

## Inner-selected complaint margin experiment

The training partition contains an inner split of 49700 training and 12426 validation messages. TF-IDF is fitted on inner training only; normalized group overlap is 0.
The candidates are averaged SGD hinge, averaged SGD modified Huber, and balanced linear SVM at C=1. All use hybrid word/character features. Earlier word-only SVM results were weaker; this comparison uses the combined representation.

| Head | Inner clean/noisy mean macro-F1 |
|---|---:|
| sgd_hinge | 0.2368 |
| modified_huber | 0.2235 |
| linear_svc | 0.2466 |

Inner-selected head: **linear_svc**. After refitting on the full original training split, original validation mean macro-F1 was **0.2292**, versus **0.2579** for the baseline. Promotion rule passed: **False**.
Not evaluated: selected candidate did not pass the validation improvement rule.
The original validation corpus was used in earlier experiments. The inner split limits reuse during this search but does not make the existing validation/test corpus fresh independent evidence. Softmax-converted margins are ranking scores, not calibrated probabilities.

## Frozen complaint semantics and fixed ensemble

The semantic classifier uses fixed pretrained multilingual representations. C=0.01 or 0.1 is selected within the training partition, with scalers fitted on inner training only. The selected head is refitted on full training before validation.
The ensemble uses a predeclared 50/50 average with the baseline. Blend weights are not tuned. Component class columns are explicitly aligned.

| Candidate | Clean validation macro-F1 | Typo validation macro-F1 | Mean | Promotion passed |
|---|---:|---:|---:|---|
| semantic | 0.1614 | 0.1434 | 0.1524 | False |
| ensemble | 0.2125 | 0.2002 | 0.2064 | False |

semantic: Not evaluated: validation improvement rule failed.

ensemble: Not evaluated: validation improvement rule failed.

All of these runs use an already explored development corpus. The bounded search and frozen encoder reduce some overfitting risks; fresh independent labels are still needed to confirm transfer.

## Bounded encoder continuation

The one-epoch checkpoint is continued for at most two further epochs, with learning rate 2e-5, weight decay 0.01, effective batch 32 and a 192-token limit. Training uses square-root inverse-frequency loss weights and typo augmentation.
Mean clean/noisy validation macro-F1 is checked after each epoch. A gain below 0.005 stops training; the best qualifying checkpoint is retained. The optimizer and scheduler restart for this continuation.

| Total epoch | Training loss | Clean/noisy validation mean F1 | Checkpoint improved |
|---|---:|---:|---|
| 2 | 2.0233 | 0.2037 | True |
| 3 | 1.8408 | 0.2192 | True |

Selected total epoch: **3**. Validation mean: **0.2192**, versus baseline **0.2579**. Promotion rule passed: **False**.
Not evaluated: bounded continuation did not pass validation promotion rule.
The original validation set has been reused; this bounded budget and stopping rule reduce some risks but do not establish absence of overfitting or independent generalisation.

## Training-internal weighting and NB-ratio comparison

A predeclared 70/15/15 split of the training partition separates fitting, candidate selection and confirmation. The selected candidate is refitted before each subsequent assessment. Fitted vocabularies and class-specific feature ratios exclude their assessment partitions.
The candidates compare balanced SGD, normalized square-root class weighting, a fixed lower-regularization variant, and one-versus-rest linear SVMs with binary word/bigram NB log-count ratios.

| Candidate | Selection clean/noisy mean F1 | Selection accuracy |
|---|---:|---:|
| reference | 0.2634 | 0.4839 |
| sqrt_weights | 0.2501 | 0.5083 |
| sqrt_less_regularized | 0.2470 | 0.4934 |
| nb_ratio_svm | 0.1698 | 0.4254 |

Selected candidate: **reference**. Final status: **selection_gain_not_confirmed**. Promotion passed: **False**.
No additional official-test evaluation was run because an earlier gate did not qualify the candidate.

The revised criteria were recorded before training in `complaint_improved_plan.json`. Confirmation and existing-validation require a mean F1 gain of 0.01, nonnegative clean F1 gain, and accuracy/typo-F1 declines no larger than 0.005. Release additionally requires positive benchmark F1 gain and a positive paired-bootstrap lower bound. The benchmark remains development-exposed; independent confirmation is still required.

## Label-supervised word/character network

This experiment learns hashed word/bigram and character embeddings jointly with a nonlinear classifier. It uses a six-epoch cap, dropout 0.4, AdamW weight decay 0.01, inverse-frequency loss weights and training-only typo augmentation.
Epoch selection uses the fitting/selection partitions from the preceding experiment; a qualified network would require a fresh refit and separate confirmation before existing validation or test evaluation.

| Epoch | Training loss | Selection clean/noisy mean F1 | Selection accuracy |
|---|---:|---:|---:|
| 1 | 2.9044 | 0.1576 | 0.3595 |
| 2 | 2.0807 | 0.1730 | 0.3772 |
| 3 | 1.4036 | 0.1866 | 0.4051 |
| 4 | 0.9998 | 0.1905 | 0.4294 |
| 5 | 0.8029 | 0.1919 | 0.4123 |
| 6 | 0.6811 | 0.1849 | 0.4010 |

Selected epoch: **5**. Final status: **selection_gate_failed**. Promotion passed: **False**.
Selected-network mean F1: **0.1919**, compared with reference **0.2634** on the same selection partition.
No additional test evaluation was performed. Declining training loss without sufficient held-out improvement does not justify deployment.

Default inference uses validation-selected backends:

- call_scam: **baseline** (primary family-held-out macro-F1 0.6308; legacy exact-template validation selection score 1.0000).
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

## Limitations

- English and Hinglish label vocabularies differ; cross-language transfer is not independently validated.
- Complaint classification has no benign class; call `none` does not prove safety.
- Rule-based PII masking is incomplete; unstructured names, addresses and contextual identifiers may remain.
- Confidence scores and review threshold are uncalibrated; unknown scams can receive high scores.
- Final reports supersede internal-test exploratory artifacts under `superseded/`.
- No RAG layer, live-call integration, or automatic complaint submission is included.
