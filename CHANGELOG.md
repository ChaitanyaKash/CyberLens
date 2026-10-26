# Changelog

## 2026-10-26 (scam/not-scam classifier and interfaces)

- Added local Streamlit interfaces: a simple layout (`app.py`) and a product-page layout (`app_saas.py`), with plain-language verdicts, safety tips and the national helpline.
- Restricted the call model's police verdict to confident matches whose text contains police or authority wording.
- Added a walkthrough notebook for the complaint and call classifiers.
- Added a binary scam/not-scam classifier for English, Hinglish and Hindi SMS-style messages (`cyberlens/scam_binary.py`, `cyberlens/scam_binary_v2.py`), with grouped splits and a template-grouped stress set. v2 reduced false alarms on unseen legitimate bank and OTP alerts from 56.5% to 0.8%.
- The product-page interface now uses this classifier as the sole scam/not-scam decision, with confidence-graded wording, a disclaimer and safeguards that prevent incident descriptions from being shown as "not a scam".

## 2026-10-21 (complaint model update)

- Added the `official` complaint training protocol, which trains on all of `train.csv` without consulting `test.csv`, and keeps plurality labels for repeated training templates with at least 60% label agreement.
- Added the `adjusted` complaint backend: hybrid TF-IDF SGD with a validation-selected class-prior correction, refitted on training plus validation.
- Promoted `adjusted` as the default complaint model under a predeclared rule. Official-test macro-F1 0.2345 to 0.2620, accuracy 0.5014 to 0.5512; novel-template macro-F1 unchanged within uncertainty.
- Added reporting for test templates absent from `train.csv` and paired template-bootstrap intervals against the previous default.

## 2026-10-21

### Classification pipeline

- Added call-scam and complaint-category classifiers with task-specific label spaces.
- Added conservative English/Hinglish normalization and partial masking of structured sensitive identifiers.
- Implemented word, character and hybrid TF-IDF baselines, custom Skip-gram negative sampling, character-subword embeddings and multilingual encoder experiments.
- Added a local desktop interface, command-line tools, package configuration and GitHub Actions checks.

### Evaluation

- Added duplicate-aware preparation, complaint train/test overlap removal and explicit handling of contradictory call labels.
- Added five-fold template-family evaluation after the exact-deduplicated call split produced optimistic results.
- Added nested family-CV selection, synthetic typo evaluation, paired-family bootstrap uncertainty and validation-gated model promotion.
- Corrected validation metrics to exclude labels present only in the test taxonomy.
- Added aggregate class-support, label-conflict and confusion analysis.

### Model comparisons

- Added a predeclared training-internal weighting/NB-ratio comparison and a label-supervised word/character network, with separate selection and confirmation boundaries, accuracy retention and final paired-bootstrap release criteria.
- Rejected both new experiments at selection. Square-root weighting improved selection accuracy but reduced macro-F1; the supervised network's falling training loss did not transfer to sufficient held-out F1. No additional official-test evaluation was performed.

- Nested call candidate selection reached clean macro-F1 0.8256 and typo macro-F1 0.7709, compared with 0.6308 and 0.6337 for the reference. The improvement interval includes zero, so the candidate remains experimental.
- Regularized, margin, frozen-semantic and fixed-ensemble complaint candidates did not exceed the sparse baseline's validation criterion.
- Bounded transformer continuation improved validation mean macro-F1 from 0.1659 to 0.2192, below the baseline's 0.2579. No additional official-test evaluation was run for rejected candidates.
- Retained the complaint baseline with official-test macro-F1 0.2345. The additional comparisons did not establish an English test improvement.

### Inference and verification

- Added experimental-model status, class-level evidence and mandatory review for rejected optional models.
- Added saved-model loading, masking, split-isolation, metric-boundary, class-order and promotion-rule checks.
- Expanded reproducible experiment reports and technical documentation.
