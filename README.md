# CyberLens

[![Core tests](https://github.com/ChaitanyaKash/CyberLens/actions/workflows/ci.yml/badge.svg)](https://github.com/ChaitanyaKash/CyberLens/actions/workflows/ci.yml)

CyberLens classifies English and Hinglish text into call-scam scenarios and cybercrime complaint categories. The pipeline combines conservative preprocessing, partial sensitive-data masking, typo-tolerant text features and task-specific classifiers.

The project compares sparse text models, a PyTorch implementation of Skip-gram with negative sampling, character-subword embeddings and multilingual transformer approaches. Model selection accounts for duplicate templates, label conflicts, class imbalance and spelling noise.

```mermaid
flowchart LR
    A[English or Hinglish text] --> B[Normalization and partial sensitive-data masking]
    B --> C[Task-specific text representation]
    C --> D[Call-scam classifier]
    C --> E[Complaint-category classifier]
    D --> F[Ranked predictions and review status]
    E --> F
```

## Results

<!-- RESULTS SUMMARY START -->
Call classification uses template-family evaluation. Complaint classification uses the supplied test set. Scores from these protocols are not directly comparable.

| Task | Backend | Evaluation protocol | Macro-F1 | Typo macro-F1 | Accuracy |
|---|---|---|---:|---:|---:|
| call_scam | baseline | 5-fold template-family CV (frozen baseline) | 0.6308 | 0.6337 | 0.7416 |
| complaint | adjusted | Supplied test.csv (official-train protocol) | 0.2620 | 0.2573 | 0.5512 |

[Full comparisons and failure cases](reports/RESULTS.md). Scores are fractions, not percentages.

On complaint test messages whose normalized template does not occur in `train.csv`, macro-F1 is **0.2362** versus **0.2376** for the previous default and accuracy **0.4962** versus **0.4745**. Most of the full-test macro-F1 gain comes from recurring complaint templates; see the results for the breakdown.

The experimental `robust` call model reached **0.8256** clean / **0.7709** typo macro-F1 in nested family CV. It remains experimental because the paired family-bootstrap interval for clean improvement includes zero. Independent confirmation is required.
<!-- RESULTS SUMMARY END -->

The initial exact-deduplicated call split produced macro-F1 of 1.0000. Related templates still crossed that split, so the primary result uses the stricter template-family protocol. Earlier scores remain in the evaluation report for comparison.

The default complaint model (`adjusted`) trains on the whole supplied `train.csv` without consulting `test.csv` and applies a validation-selected class-prior correction to the hybrid TF-IDF classifier. Compared with the previous default, official-test macro-F1 rises from 0.2345 to 0.2620 and accuracy from 0.5014 to 0.5512. The gain is concentrated in subtypes dominated by recurring complaint templates; on test templates absent from `train.csv`, macro-F1 is unchanged within uncertainty and accuracy improves by about two points. The [results](reports/RESULTS.md) give the per-slice and per-class breakdown.

Complaint classification remains challenging. Training support ranges from 23 to 15,162 examples per class. Three test labels have no trained class, and 130 normalized test templates carry conflicting labels. These cases remain in the reported test evaluation.

## Tasks and datasets

| Task | Training source | Label space |
|---|---|---|
| `call_scam` | `India_Cyber_Scam_Hinglish_Dataset.csv` | `none`, `police_digital_arrest`, `police_blackmail`, `other_scam` |
| `complaint` | `train.csv` | 39 category/subcategory combinations |

`test.csv` is the supplied complaint test set. The default `official` training protocol uses all of `train.csv` and never reads `test.csv` during training. The historical protocol excluded training templates that also occur in `test.csv`; its models and reports are retained for comparison. Test messages whose template occurs in `train.csv` are reported separately from those that do not.

The Hinglish dataset contains 10,000 rows but only 743 distinct texts. Identical messages carry conflicting labels among `aadhaar`, `amazon`, `bank_kyc`, `lottery` and `relative`; these labels map explicitly to `other_scam`. The mapping reduces subtype granularity rather than establishing equivalence between the underlying scam mechanisms.

The datasets use different taxonomies and are evaluated as distinct tasks. Both classifiers accept English and Hinglish strings, but reliable cross-language transfer is not independently established. Complaint labels contain no benign class; a complaint-category prediction does not establish that a message is fraudulent.

## Installation

Python 3.12 is recommended. Sparse models run on CPU; neural experiments require the optional dependencies, and transformer fine-tuning expects CUDA.

```powershell
git clone https://github.com/ChaitanyaKash/CyberLens.git
cd CyberLens
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
# Optional neural dependencies; install an appropriate CUDA PyTorch build.
.\.venv\Scripts\python -m pip install -e '.[neural]'
```

Datasets and trained weights are not distributed in the repository. Place the three dataset files in `data/raw/`, then run preparation and training. Public CI runs self-contained tests; integration tests requiring private datasets, local models or optional neural dependencies skip when those assets are unavailable.

## Usage

```powershell
.\run.ps1 prepare
.\run.ps1 train --backend baseline
# Default complaint model: official-train protocol with class-prior correction
.\run.ps1 prepare --protocol official
.\run.ps1 train --backend adjusted --task complaint

# Desktop interface
.\classify.ps1

# Command-line classification
.\run.ps1 predict --text 'police bol rha hu abhi paise bhejo'
.\run.ps1 predict --task complaint --text 'Someone sent me a fake UPI payment request'
.\run.ps1 predict --interactive

# Experimental call model; requires training and local encoder weights
.\run.ps1 predict --backend robust --task call_scam --text 'police bol rha hu abhi paise bhejo'
```

PowerShell launchers resolve Python through `CYBERLENS_PYTHON`, a project `.venv`, an available NLP conda environment, or `python`. The `cyberlens` console command is also available after installation.

### Local UI

A browser interface for classifying new complaints and messages runs entirely on the local machine:

```powershell
.\.venv\Scripts\python -m pip install -e '.[ui]'
.\run_ui.ps1        # simple layout: streamlit run app.py, then open http://localhost:8501
.\run_ui_saas.ps1   # product-page layout: streamlit run app_saas.py, then open http://localhost:8502
```

It gives a plain-language verdict that combines the complaint and call-scam models, with safety tips and the national cybercrime helpline. A collapsed details section shows the top three labels of each model, the masked text, timing and the sparse model's n-gram evidence. The champion fusion model is used when `models/complaint_fusion.joblib` is present; fast mode uses the sparse `adjusted` model on CPU. The complaint confidence threshold is read from `reports/complaint_selective.json` when available. Usage statistics collection is disabled in `.streamlit/config.toml`, and input text is not stored.

In the product-page layout, a separate scam/not-scam classifier (TF-IDF v2) decides whether a message is a scam. Messages it considers legitimate are shown as "Not a scam", "Probably not a scam" or "Unclear" with a disclaimer; the complaint model's closest category remains available but collapsed. Descriptions of an incident, texts naming the offence and texts under four words are never shown as "not a scam". Train the classifier with:

```powershell
python -m cyberlens.scam_binary_v2 tfidf     # TF-IDF v2, CPU
python -m cyberlens.scam_binary_v2 encoder   # optional xlm-roberta-base, CUDA
python -m cyberlens.scam_binary_v2 report
```

The raw datasets are expected under `data/raw/scam_binary/`; see [the results](reports/RESULTS.md#scam-vs-not-scam-message-classifier).

Predictions include ranked alternatives, class-level evaluation evidence and a review flag. Scores are uncalibrated. An experimental model that has not passed its promotion rule always requires review. Weak complaint-subtype evidence can produce a broader category suggestion.

## Experiments

| Backend | Representation and classifier |
|---|---|
| `baseline` | Word, character and hybrid TF-IDF with averaged log-loss SGD |
| `word2vec` | Custom Skip-gram negative sampling; mean document vectors and a linear head |
| `subword` | Skip-gram with hashed character n-gram representations |
| `transformer` | Multilingual DistilBERT fine-tuning |
| `hierarchical` | Conditional financial-subtype expert combined with baseline scores |
| `robust` | Nested family-CV comparison including frozen multilingual features |
| `regularized` | Fixed Naive Bayes and logistic-head comparison for complaints |
| `margin` | Inner-selected hinge, modified Huber and linear-SVM heads |
| `semantic` | Frozen multilingual features with an inner-selected regularized head |
| `ensemble` | Fixed equal-weight sparse/semantic score average |
| `continued` | Bounded continuation of the one-epoch complaint transformer |
| `improved` | Training-internal weighting and NB-ratio feature comparison with confirmation gates |
| `supervised` | Jointly learned word/character embedding bags and a regularized nonlinear head |

Examples:

```powershell
.\run.ps1 train --backend word2vec --epochs 3 --max-pairs 300000
.\run.ps1 train --backend subword --epochs 3 --max-pairs 300000
.\run.ps1 train --backend hierarchical --task complaint
# Public encoder weights are required for neural experiments.
.\run.ps1 train --backend transformer --task call_scam --epochs 3
.\run.ps1 train --backend transformer --task complaint --epochs 1 --batch-size 32 --accumulation 1
.\run.ps1 train --backend robust --task call_scam
.\run.ps1 train --backend regularized --task complaint
.\run.ps1 train --backend margin --task complaint
.\run.ps1 train --backend semantic --task complaint
.\run.ps1 train --backend ensemble --task complaint
.\run.ps1 train --backend continued --task complaint
.\run.ps1 train --backend improved --task complaint
.\run.ps1 train --backend supervised --task complaint
python -m cyberlens.report
python -m unittest discover -s tests -v
```

The ensemble requires trained baseline and semantic artifacts. Continuation requires the saved one-epoch complaint transformer. The supervised experiment requires the completed `improved` reference comparison to verify matched training-internal partitions. Experiments rejected before full-training refit do not create inference artifacts. Full reproduction, including public encoder acquisition when required, is available through `reproduce.ps1`.

The default backend follows the recorded promotion criteria. Historical complaint comparisons require a clean/noisy validation macro-F1 gain of at least 0.02 before an additional official-test evaluation. The `improved` experiment instead uses separate training-internal selection and confirmation partitions, followed by a validation gain of at least 0.01 and accuracy/robustness constraints. Release also requires a positive paired-bootstrap lower bound on the final test gain. Its complete criteria are recorded before training in `reports/complaint_improved_plan.json`. The call candidate has its own family-held-out uncertainty criterion. Validation has been reused during development; independent evaluation remains necessary.

## Documentation

- [Technical design and methodology](TECHNICAL_GUIDE.md)
- [Evaluation results](reports/RESULTS.md)
- [Aggregate complaint error analysis](reports/complaint_error_audit.json)
- [Changelog](CHANGELOG.md)

## Limitations

Structured masking covers selected URL, email, phone, payment and identifier patterns. Names, addresses and contextual identifiers may remain; masked text is not guaranteed anonymous. Processing and inference run locally. Raw datasets and model artifacts are excluded from version control; only trusted serialized models should be loaded.

Synthetic typo tests cover a limited corruption pattern. Template-family grouping is heuristic. Performance on new scam scenarios, languages and label taxonomies requires independent data. The current scope is text classification; retrieval-based guidance and report drafting are planned extensions.
