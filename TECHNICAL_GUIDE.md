# Technical Design and Evaluation

## Problem definition

CyberLens predicts incident categories from English or Hinglish text, including spelling variation. Two task-specific taxonomies are maintained because the source datasets do not provide an aligned multilingual label space.

| Task | Scope | Supported training classes |
|---|---|---:|
| Call scams | Call-scam scenarios and a source-defined benign class | 4 |
| Complaints | Cybercrime parent category and subtype | 39 |

A classifier accepting both languages does not establish cross-language reliability. Similarly, the absence of a benign complaint class prevents complaint predictions from serving as a general scam-versus-legitimate decision.

## Data preparation and label quality

The Hinglish source has 10,000 rows and 743 distinct texts. Five source labels assign different outcomes to identical messages: `aadhaar`, `amazon`, `bank_kyc`, `lottery` and `relative`. These labels are mapped to `other_scam`; the remaining classes are `none`, `police_digital_arrest` and `police_blackmail`. This is an explicit reduction in subtype resolution, not a semantic equivalence claim.

Complaint training uses `train.csv`; `test.csv` provides the official test set. Normalized training templates overlapping the supplied test are excluded from training. Conflicting training groups are excluded, duplicate training groups are consolidated, and classes with insufficient training support are recorded in the data audit. The prepared complaint partitions contain 62,126 training, 15,532 validation and 30,771 test messages.

The official test retains its repeated templates and label disagreements. Three labels in that test have no trained class. They count as unsupported outcomes in test metrics, but do not add zero-support classes to validation macro-F1. The supported training taxonomy contains 39 classes; the test evaluation includes its additional labels.

### Class imbalance and ambiguity

Training support ranges from 23 to 15,162 messages per class, approximately 659:1. Equal class weighting in macro-F1 makes poor rare-class performance visible rather than allowing frequent financial complaints to dominate the metric.

The official test has 130 normalized templates with conflicting labels. Distribution differences also matter: one sexual-abuse-content class has 186 training messages and 911 test messages. These observations identify evaluation and data-quality challenges without establishing a single cause for every classification error.

A training-only diagnostic covers 2,886 cyberattack messages across seven subtypes. Class-average word/bigram TF-IDF vectors have pairwise cosine similarities of 0.9923–0.9983. Similar average wording warrants review of subtype evidence; it does not prove that labels are incorrect or prediction is impossible. No automatic relabeling follows from this diagnostic.

## Processing pipeline

```text
Text input
  -> Unicode and whitespace normalization
  -> Partial masking of structured sensitive identifiers
  -> Task-specific representation
  -> Trained classifier
  -> Ranked scores and class-level evaluation evidence
  -> Review status and optional broad-category suggestion
```

Training additionally performs label auditing, grouping, partition construction, training-only feature fitting, validation selection and held-out assessment. Trained artifacts and aggregate reports provide reproducible inference and evaluation.

### Why global spelling correction is excluded

English dictionary correction can alter valid Hinglish words. Romanized Hindi also permits several valid spellings of the same word. The pipeline therefore preserves wording, negation and useful context rather than enforcing an English-only vocabulary.

Character n-grams provide tolerance to local spelling changes. Word n-grams retain phrases and contextual clues. Subword embeddings and multilingual tokenization offer additional representations for spelling variants and unseen tokens. None guarantees semantic equivalence between every variant.

Stemming and aggressive stop-word deletion are excluded from the primary pipeline because they can remove distinctions needed for threats, negation and payment context.

## Sensitive-data masking

The masker identifies selected structured spans and replaces their values with typed placeholders, such as `<PHONE>`, `<EMAIL>`, `<URL>`, `<UPI_ID>` and `<OTP>`. Retaining entity types preserves useful context while reducing direct identifier exposure.

For example, an OTP value can be masked while the surrounding request to share an OTP remains available to the classifier. Context-sensitive rules distinguish some financial identifiers from ordinary amounts.

Span detection precedes replacement so overlapping detections can be resolved against character offsets. Masking is tested for idempotence and preservation of typed placeholders during synthetic corruption.

This is partial masking. Names, addresses, contextual identifiers and unsupported formats may remain. Candidate identifier matches are not identity verification, and masked text is not guaranteed anonymous. Optional externally supplied entity spans provide an integration point for a separately evaluated local NER component.

## Sparse text baseline

The baseline compares three representations:

| Representation | Configuration |
|---|---|
| Words | Unigrams/bigrams; minimum document frequency 2; at most 65,000 features |
| Characters | `char_wb` 3–5-grams; minimum document frequency 2; at most 100,000 features |
| Hybrid | Concatenated word and character features |

TF-IDF uses sublinear term frequency. Vocabularies and IDF values are learned from the relevant training partition. The classifier is averaged SGD with log loss, balanced class weights, alpha `1e-5`, an 80-iteration cap and seed 42.

Averaging reduces sensitivity to individual parameter updates. Balanced weights increase the contribution of minority classes, but can also increase false positives. Neither mechanism replaces validation.

The selected representation maximizes the mean of clean and synthetic-typo validation macro-F1. Sparse features provide an efficient reference and expose whether more complex representations add measurable value.

## Skip-gram with negative sampling

The embedding implementation uses PyTorch directly rather than delegating training to a Word2Vec library. Tokenization and vocabulary construction use training text. Words below the frequency threshold are excluded from the vocabulary.

For a center word c and a positive context word o, negative sampling minimizes:

```text
-log sigmoid(u_o dot v_c)
-sum over negative words n: log sigmoid(-u_n dot v_c)
```

Separate input and output embedding tables learn center/context relationships. The recorded configuration uses 96-dimensional embeddings, a context window of four, five negatives per positive pair, minimum frequency two and a negative-sampling distribution proportional to count^0.75.

The bounded experiment samples at most 300,000 positive pairs and trains for three epochs. It does not implement frequent-word subsampling. Its results describe this configuration rather than the entire Word2Vec model family.

Document vectors average the learned token representations. A scaled logistic classifier learns task labels from those vectors. Mean pooling is inexpensive but discards order, can dilute short decisive phrases and depends on vocabulary coverage.

### Character-subword extension

The subword variant represents tokens with hashed character 3–5-grams, using 50,000 buckets and a maximum of 24 grams per token. Shared grams allow related spellings and some unseen words to share representation components.

Hashing bounds memory but introduces collisions. Subword sharing can help local spelling variation without guaranteeing that similar strings have the same meaning. The recorded complaint results remained below the sparse baseline, so these models are retained as representation-learning comparisons.

## Multilingual encoder approaches

The pretrained encoder is `distilbert/distilbert-base-multilingual-cased`. Its use is transfer learning; the transformer is not pretrained from scratch in this project.

### Frozen representation

The frozen variant disables encoder gradients and dropout. Attention-mask mean pooling excludes padding, followed by L2 normalization. A training-fitted StandardScaler and L2-regularized logistic classifier learn the task-specific boundary.

Encoder computation is fixed and may be cached before splitting classifier folds. Scaler statistics cannot be fitted globally: each inner fold learns them from its own training portion. The same restriction applies to learned text vocabularies and IDF values.

### Fine-tuning

Sequence classification uses the multilingual encoder with a task-specific head. Training includes square-root inverse-frequency loss weights, AdamW, gradient clipping, bfloat16 computation, gradient checkpointing and a maximum length of 192 tokens.

Thirty percent of training messages are eligible for controlled synthetic spelling augmentation. Evaluation typo inputs are constructed independently from their clean evaluation messages; validation and test messages are never added to training.

The bounded complaint continuation begins from the one-epoch checkpoint and permits at most two additional epochs. It uses learning rate `2e-5`, weight decay `0.01`, microbatch 16 and two-step gradient accumulation, for effective batch 32. The optimizer and warmup/decay scheduler restart for this continuation.

After each epoch, mean clean/noisy validation macro-F1 must improve by at least 0.005 to retain the checkpoint and continue. The maximum budget is three total epochs. Test scores cannot select a checkpoint.

## Evaluation protocols

### Exact-template control

Normalized fingerprints keep exact templates within one partition. This reduces direct duplicate leakage but does not control all semantic paraphrases or template variants.

The initial exact-deduplicated call split produced perfect scores. These estimates were optimistic because related scripts still crossed the partition boundary. They remain documented as a protocol limitation rather than a production-performance claim.

### Template-family assessment

Character TF-IDF cosine similarity at threshold 0.80 defines links between call messages. Connected components form 77 families. Five-fold StratifiedGroupKFold keeps each family on one side of an assessment boundary.

Family construction uses unsupervised grouping features; those features are not passed to the classifier. Each classifier fold fits its own vocabulary and parameters on its training families. The grouping is heuristic and is not a human-verified semantic benchmark.

The fixed word baseline reaches clean macro-F1 0.6308 and typo macro-F1 0.6337 under this protocol, substantially below its easy-split score.

### Nested selection

The call improvement experiment fixes six candidates: the word reference; word, character and hybrid logistic models at C=1; and frozen multilingual features with C=0.01 or 0.1.

Each of five outer folds uses three inner family folds for selection. Inner selection maximizes mean clean/noisy macro-F1. Only the inner-selected candidate is refitted on the outer training portion and assessed on outer messages. Each message receives one outer prediction, and runtime checks reject overlapping families or training folds missing supported classes.

The selected procedure reaches clean macro-F1 0.8256 and typo macro-F1 0.7709. Its final inference head is selected through a training-corpus grouped search and refitted on the development corpus. That refitted artifact is distinct from the models producing outer-fold predictions.

### Uncertainty and promotion

The call promotion rule requires a mean clean/noisy gain of at least 0.02, a positive lower bound for clean improvement, and no typo-F1 decline greater than 0.005.

Five hundred paired bootstrap resamples operate on whole families, with identical resampled weights for the reference and candidate. The approximate 95% interval for clean F1 improvement is [-0.0283, 0.4064]. Since it includes zero, the candidate remains experimental despite the higher point estimate.

The interval is approximate. Few independent families, overlapping CV training portions and heuristic grouping limit its interpretation. Earlier corpus exploration also prevents treating this as fresh external confirmation.

### Complaint selection

Additional complaint candidates require an original-validation clean/noisy mean macro-F1 gain of at least 0.02 over the sparse baseline before another official-test evaluation.

The margin and frozen-semantic searches use a stratified 80/20 inner split of the training partition. Inner feature fitting and regularization selection exclude the existing validation set. This reduces direct reuse within each search but cannot erase prior development exposure.

Synthetic typo evaluation swaps internal letters in 25% of eligible words. Placeholders and numeric values are protected. This tests one controlled corruption pattern, not the full distribution of real spelling errors.

## Additional comparisons and outcomes

| Complaint candidate | Clean/noisy validation mean macro-F1 | Outcome |
|---|---:|---|
| Sparse baseline | 0.2579 | Retained |
| Best Naive Bayes/logistic comparison | 0.2472 | Promotion criterion not met |
| Inner-selected hybrid margin model | 0.2292 | Promotion criterion not met |
| Frozen multilingual head | 0.1524 | Promotion criterion not met |
| Fixed sparse/semantic ensemble | 0.2064 | Promotion criterion not met |
| Three-total-epoch encoder | 0.2192 | Promotion criterion not met |

The regularized comparison includes Complement Naive Bayes at alpha 0.5 and one-versus-rest logistic regression at C=1 with and without balanced weights. The margin comparison includes averaged hinge SGD, averaged modified Huber SGD and balanced linear SVM at C=1. Modified Huber reached its fixed iteration cap in the recorded inner fit; this limits conclusions about that configuration.

The ensemble uses a predeclared equal-weight average. Component class vocabularies must match, and score columns are explicitly aligned before averaging. Scores remain uncalibrated; averaging different representations does not guarantee improvement.

The financial expert redistributes the baseline's financial-category probability mass across seven conditional subtypes while preserving other categories. Its validation result did not justify promotion.

Encoder continuation improved its own validation mean from 0.1659 to 0.2037 and then 0.2192. This supports additional fitting within that model, but the sparse baseline remained stronger. Rejected complaint candidates did not receive new official-test evaluations. The baseline's official-test macro-F1 remains 0.2345; the additional comparisons did not establish an English test improvement.

## Metrics and inference behavior

Macro-F1 gives every evaluation class equal weight. Weighted F1 scales class contributions by support, while accuracy measures the fraction of correct predictions. These metrics are not interchangeable. The complaint baseline's 0.5014 accuracy and 0.5081 weighted F1 do not replace its 0.2345 macro-F1.

Inference returns ranked alternative labels, uncalibrated scores, class-level evaluation evidence and review status. A class-level F1 is not the probability that an individual prediction is correct.

Review heuristics include low score, insufficient feature coverage, class F1 below 0.5, support below 20 and experimental status. Complaint subtype disagreement with aggregated parent-category scores can also trigger a broad-category suggestion. These are triage heuristics, not calibrated safety guarantees.

A model that fails its promotion rule remains optional and always requires review. The default backend is chosen through the recorded selection policy, rather than switching to whichever model has the most complex architecture.

## Verification and reproducibility

Tests cover structured masking, negation preservation, idempotence, placeholder-safe corruption, partition isolation, validation/test vocabulary boundaries, embedding gradients, serialized backend loading, frozen encoder state, margin ranking and ensemble class alignment.

The source contains command-line training interfaces and a dependency-ordered reproduction script. Aggregate JSON reports retain confusion matrices, class metrics, configurations and selection outcomes. Datasets, vector caches and model weights are not distributed in the repository.

## Further development

The next evaluation milestone is an independently collected, reviewed English/Hinglish set with aligned labels, diverse scenarios, legitimate warnings and realistic spelling variation. Rare classes and subtypes with near-identical wording require particular attention.

A future RAG layer can retrieve reviewed guidance using the message and classification evidence. Retrieval must preserve uncertainty rather than turn a tentative category into a fact. Report drafting should distinguish supplied evidence from missing information and retain user review before external use.

<!-- GENERATED RESULTS -->

## Evaluation reference

[Evaluation results](reports/RESULTS.md) contain the measured comparisons, class-level metrics and model-promotion decisions.
