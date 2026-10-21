# What changed

## 2026-10-21 — Bounded improvement experiments

I added a six-candidate call experiment with five outer and three inner template-family folds. The encoder stays frozen, and vocabularies/scalers/classifiers fit their own training folds. Nested clean macro-F1 improved from 0.6308 to 0.8256; typo macro-F1 improved from 0.6337 to 0.7709.

I kept that model experimental because the approximate paired family-bootstrap interval for clean improvement includes zero. The default-promotion rule was fixed before the run; I did not change it to accept a higher point estimate. The optional `robust` backend is available in the CLI and desktop interface.

I also added a fixed three-head complaint comparison using the same training-only TF-IDF features. It only evaluates the selected head on the supplied test if the clean/noisy validation mean improves by at least 0.02. The generated results record the outcome, including rejected candidates. I expanded reproduction commands, inference checks and the interview notes to explain the mechanism, uncertainty and remaining need for independent data.

## 2026-10-21 — First working text-classification pipeline

I moved the experiments into a separate project so I could work without changing my original notebooks or datasets.

- Added separate classifiers for call-scam scenarios and cybercrime complaint categories. The two datasets use different labels, so I kept their outputs separate.
- Added conservative preprocessing and partial masking for structured identifiers. I left English/Hinglish words alone instead of running a global spellchecker.
- Added word, character and combined TF-IDF baselines.
- Implemented Skip-gram with negative sampling in PyTorch, then added a character-subword version to compare spelling and unseen-word handling.
- Added a multilingual DistilBERT fine-tuning path after the initial English complaint results were weak.
- Added a local desktop interface and terminal commands for training and prediction.
- Added data audits, clean/noisy evaluation, saved models, source-file integrity checks and an interview guide.
- Added package metadata and GitHub Actions tests. Raw complaints, old notebook outputs, caches and trained weights stay out of the repository.

The biggest change was in how I handled the data. The Hinglish file contains only 743 unique messages, and five labels conflict on identical text. I merged those labels into `other_scam`. For English complaints, I used the supplied `test.csv` and removed overlapping templates from training. The reports explain these choices and keep the weaker results visible.

The encoder comparison and final evaluation are recorded in [the results](reports/RESULTS.md). I’ll use this file to keep track of later fixes and experiments too.

### Evaluation fixes after the first push

The initial call split had no normalized duplicate overlap, but that was not enough: related template variants still crossed the split. I added a five-fold evaluation that keeps 77 near-duplicate families together. Macro-F1 fell from 1.0000 on the easy split to 0.6308 on this harder check. I changed the README headline and kept the earlier result only for comparison.

I also fixed a reporting issue: three complaint labels occur only in the supplied test file. They should count as unsupported classes in test results, but should not add zero-support classes to the validation macro-F1 denominator. The fix recomputes metrics from saved confusion counts without changing predictions or fitting on test data.

I tried a dedicated financial-fraud expert as well. It did not improve the combined clean/noisy validation score, so I kept the simpler baseline as the default and saved the experiment for comparison. The app now shows class-level evaluation evidence and marks weak subtype suggestions for review rather than treating a high model score as sufficient evidence.

I checked every saved backend in a fresh inference path and fixed the custom estimator's module entry point so its model can be loaded outside the training command. All 13 local tests pass, including model loading, split checks and the validation-report regression test.
