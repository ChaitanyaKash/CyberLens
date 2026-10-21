# What changed

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
