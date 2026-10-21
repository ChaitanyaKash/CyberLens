# My CyberLens build notes

I started with a preprocessing problem: how do I handle English and Hinglish together without an English spellchecker damaging the Hindi words? That led me into a bigger issue with the datasets themselves. These notes explain what I tried, what I changed, and what I still need to fix. I’m keeping the mechanisms here so I can explain the project properly instead of just running the code.

The numbers come from the saved experiment reports. When I say I considered an approach, that does not mean I trained it. I kept the original project untouched and did the work in a separate copy.

## 1. What I want this version to do

Given a suspicious English or Romanized Hindi-English message, predict a useful incident category despite spelling noise. The first milestone is text classification. Audio transcription, live-call capture, report generation and RAG are separate future components.

The datasets describe two different tasks:

1. Short Hinglish call-like messages with scam/non-scam and scenario labels.
2. English cybercrime complaints with parent categories and subcategories.

I kept two outputs because the datasets use different definitions of a scam type. Forcing them into one label list would mean inventing relationships I cannot justify from the data. The app therefore contains separate classifiers for call scenarios and complaint categories. Both accept English and Hinglish input, but I still need independent language-specific tests before claiming that every category works equally well in both languages.

## 2. Source preservation and evidence trail

- `data/raw/train.csv`: original complaint training source, copied byte-for-byte.
- `data/raw/test.csv`: official complaint test supplied later in the session, copied byte-for-byte.
- `data/raw/India_Cyber_Scam_Hinglish_Dataset.csv`: second training source.
- Stemmed train/test CSVs are preserved but not used: they are derived data and could duplicate source examples or remove useful language clues.
- `reference/`: copies of notebooks, scripts, earlier UI files, PDF reports, plan and flowchart.
- `reports/source_integrity.json`: compares SHA-256 hashes of copied files and their original paths.
- `reports/environment.json`: actual Python, package and CUDA versions.

The launchers reuse the existing NLP conda environment without installing or changing packages. Models and report artifacts are stored in this new folder. Raw/reference data is excluded by `.gitignore`; Git ignore is convenience, not access control or encryption.

## 3. The most important discovery: label/data problems

### 3.1 Hinglish duplicates and contradictions

There are 10,000 source rows but only 743 distinct texts. A random row split would let the classifier see copies of test messages during training and produce misleading metrics.

More seriously, identical texts have different labels among `aadhaar`, `amazon`, `bank_kyc`, `lottery`, and `relative`. There are 105 conflicting normalized templates before the mapping. A deterministic text-only classifier cannot infer which of these mutually conflicting labels is correct for identical input. More GPU memory cannot restore information absent from the input.

Initial experiment: exclude all conflicting groups. This left only three call labels, losing all examples from the five affected subtype classes. That was too destructive for this milestone.

Final decision: map those five labels to `other_scam`. The other classes are `none`, `police_digital_arrest`, and `police_blackmail`. This preserves 743 unique examples and four coherent labels. The source remains unchanged. `call_label_conflicts.csv` records hashed template IDs, counts and original label combinations. The mapping is an explicit loss of granularity, not a claim that five scam mechanisms are equivalent.

To recover seven meaningful subtypes, collect or relabel messages that contain distinguishing evidence, e.g. bank KYC versus lottery fees. Do not create distinct labels from identical text using caller metadata that will not exist at text-only inference time.

### 3.2 Complaint labels

Complaint targets combine `category :: sub_category`. This keeps parent context when subcategory names are reused. Missing subcategories become `Unspecified`; the parent label is retained. A label with fewer than ten unique usable training examples is excluded from fitting and explicitly audited. Any unsupported official-test labels remain in the test metrics and count as failures.

I excluded training complaints whose normalized templates had contradictory labels. Taking the most common label would hide the disagreement rather than resolve it. Some categories also overlap in meaning: a UPI scam might involve internet banking and a fraudulent call at the same time. That ambiguity is still part of the problem.

The official complaint dataset has no benign examples. Its classifier is a conditional incident categorizer, not a scam-versus-safe detector. It will assign an incident category to a benign sentence too. The application displays this limitation.

### 3.3 Call label names are imperfect descriptions

Source inspection shows `police_blackmail` primarily contains accusations about obscene images/videos and misuse of a phone number. `police_digital_arrest` also covers broad police verification, Aadhaar/SIM misuse and arrest-related claims, rather than exclusively explicit digital-arrest scripts. Read these as **source taxonomy labels**, not legally or semantically complete definitions. The first hand-authored challenge used `police_blackmail` too broadly for relative-arrest extortion; its expected labels were corrected to match the documented source policy and two obscene-content threat examples were added. That diagnostic is therefore development data, not a pristine independent gold benchmark. Label-semantic inspection is as important as model accuracy.

## 4. Exact pipeline

```text
COPY SOURCES (no edits to originals)
  -> validate columns and missing text
  -> conservative normalization and rule masking
  -> normalized-template fingerprint
  -> task-specific label policy and conflict audit
  -> preserve official complaint test
  -> remove complaint train templates overlapping official test
  -> remove contradictory training templates
  -> deduplicate training templates
  -> audit/exclude extremely rare training classes
  -> stratified training/validation split
  -> fit features/embeddings ONLY on training
  -> fit classification head ONLY on training
  -> compare clean + synthetic-noisy validation macro-F1
  -> save selected sparse baseline
  -> evaluate official/internal held-out test and typo stress test
  -> save model, per-class metrics, confusion matrix, environment, integrity audit

INFERENCE
  -> input validation
  -> same normalization/masking function
  -> task-specific fitted representation
  -> classification head
  -> ranked category alternatives + uncalibrated scores + review heuristic
  -> local desktop/CLI output (no automatic input logging)
```

The call source uses an internal 60/20/20 split after template deduplication. The complaint source uses an 80/20 training/validation split of eligible `train.csv` examples and the supplied `test.csv` for final testing. These percentages refer to eligible deduplicated training data, not the raw source row counts.

That original call split turned out to be too easy. Exact deduplication removed identical messages, but greeting variations and related scripts still crossed the boundary. I added a harder five-fold group evaluation: character 3–5-gram TF-IDF cosine similarity at 0.80 defines near-duplicate links, and connected components form 77 families. A whole family stays on one side of each fold. Every message is evaluated once, and all four classes remain represented in training in every fold. The frozen word baseline scores about **0.631 macro-F1**, compared with 1.000 on the earlier split. The README now uses this grouped result.

I use all texts to identify related groups, but those grouping features never enter the classifier. Each fold fits its own vocabulary and classifier only on its training families. The threshold is a heuristic, not a human annotation of semantic similarity; some related paraphrases can still be missed, and chains of links can join different-looking examples. I recorded family-count sensitivity at 0.75, 0.80, 0.85 and 0.90 without choosing a threshold based on classifier accuracy. The deployed call checkpoint remains the original training artifact; the grouped score estimates the frozen baseline method across fresh fold models, not a separate independent test of that exact checkpoint.

The test file arrived after initial internal-split experiments. Those earlier artifacts were superseded; the final complaint experiments are rerun with the official test policy. `reports/superseded/` contains exploratory reports, not final comparison evidence. No test examples are added to the training vocabulary or training corpus.

## 5. What preprocessing does, and deliberately avoids

`preprocessing.py` normalizes Unicode with NFKC, removes selected zero-width artifacts, standardizes whitespace and masks selected structured identifiers. It does not translate the entire text, delete non-English words, stem words, remove negation, or run an English spellchecker.

Examples of dangerous universal mappings:

- `is`: Hindi demonstrative in `is account mein`, English verb in `bank is closed`.
- `ni`: possible shortened negation, but also a token/initial/fragment in other contexts.
- `h`: possible `hai`, but may be an initial or identifier fragment.
- `ap`: possible `aap`, but also an abbreviation.

A small opt-in normalization dictionary exists, but I left `variants=False` for the main experiments. I wanted to see how much spelling variation the representations could handle before adding correction rules. Training and inference use the same preprocessing function.

Canonical spelling is not a prerequisite for classification. The task is to preserve discriminative information; making text look grammatically clean can reduce accuracy if it changes meaning.

## 6. Sensitive-detail masking mechanism

The implemented masker is intentionally described as **partial rule-based masking**. It handles selected URLs, emails, UPI-like identifiers, PAN-like strings, Aadhaar-like candidates, Indian mobile-like numbers, and context-associated OTP/PIN/account/card values.

Steps:

1. Detect candidate spans with original character offsets.
2. Assign higher priority to more specific/contextual rules.
3. Resolve overlaps so an email is not split into a UPI ID, or an OTP value is not independently interpreted as another number.
4. Replace spans from right to left so offsets remain valid.
5. Return typed placeholders and entity counts, not raw-value logs.

Example:

```text
Before: OTP 123456 batao aur +91 9876543210 par call karo
After:  OTP <OTP> batao aur <PHONE> par call karo
```

The word `OTP` is preserved as a scam clue; only its actual value is masked. Public organization names and ordinary amounts remain useful classification signals. This is not a public-sharing policy: amounts, locations, dates and contextual details can also identify people.

Important limitations:

- Names and detailed addresses are not reliably removed by these rules.
- Number patterns can overmask non-sensitive numbers or miss unusual spacing/formats.
- `AADHAAR_CANDIDATE` means shape-based candidate, not verified Aadhaar; no claim of checksum validation.
- Reverse-order context (e.g. `123456 is my OTP`) is not fully covered.
- Plain domains without `http`/`www`, obfuscated addresses and spelled-out numbers may be missed.
- Phone-like strings embedded in UPI/email spans are handled by overlap priority, but this does not make the system exhaustive.
- Original CSVs and copied notebook outputs can contain PII. Masked text cannot be assumed anonymous.

The masker accepts optional `(start, end, kind)` spans as an integration point for a locally evaluated NER model. There is no silently downloaded or remotely invoked NER service. Next steps for a proper release are local name/address NER, manual span annotations, entity-wise precision/recall, robust format validation, and user review for exports.

## 7. Approach A: sparse TF-IDF classifiers

Three measured variants use exactly the same data split:

1. Word unigrams and bigrams.
2. Character n-grams of length 3–5 inside padded word boundaries (`char_wb`).
3. A union of word and character feature matrices.

TF-IDF gives more weight to terms that are frequent in one document but not universal across the training corpus. The implementation uses sublinear term frequency, effectively `1 + log(count)` for positive counts. Scikit-learn's smoothed IDF is based on `log((1+n)/(1+df)) + 1`, followed by default L2 normalization. Document-frequency statistics and vocabulary are learned only from training examples.

Character features share evidence across nearby spellings. `payment` and a misspelling can retain several common n-grams even if the complete word differs. This is partial robustness, not universal typo correction. `char_wb` does not explicitly model word order across boundaries; the hybrid's word bigrams restore some short context.

The classifier uses `SGDClassifier(loss='log_loss')`: a linear logistic-loss model fitted using stochastic gradient descent. In multiclass use this is an array of one-versus-rest binary classifiers, with probability outputs normalized by the library. These scores are not calibrated on an independent set. `average=True` averages learned coefficients across optimization updates. L2 regularization discourages fitting noisy isolated features; `alpha=1e-5` controls its strength.

`class_weight='balanced'` weights classes approximately inversely to their training frequencies. It helps ensure minority classes influence the loss, but can increase false positives for rare categories and does not fix incorrect labels.

Word features are capped at 65,000; character features at 100,000. Sparse matrices store nonzero values rather than dense document-by-vocabulary arrays. CPU training is appropriate here: using the GPU is not a quality metric.

The selected baseline maximizes the mean of clean validation macro-F1 and synthetic-typo validation macro-F1. Test metrics are reported afterward and never used in this selection rule. The best baseline is saved independently per task. The neural experiments remain selectable for comparison and learning.

## 8. Approach B: Word2Vec Skip-gram from scratch

My implementation is in `embeddings.py`. I used PyTorch for tensors and automatic differentiation. I implemented the vocabulary, context pairs, negative sampling, embedding tables, SGNS loss, training loop and document pooling instead of calling gensim's Word2Vec trainer. So “from scratch” here means implementing the algorithm with PyTorch; it does not mean writing an autograd engine.

Mechanism:

1. Tokenize training messages, preserving typed placeholders as atomic tokens.
2. Keep tokens occurring at least twice, with a 40,000-word cap.
3. Sample center/context pairs within a four-token context window, without crossing document boundaries.
4. Maintain separate input and output embedding tables with 96 dimensions.
5. Sample five negative context words per positive pair from `count(word)^0.75`, normalized over the training vocabulary.
6. Optimize the negative-sampling objective with Adam.

For center vector `v`, true context `u_pos`, and negative contexts `u_neg`:

```text
loss = -log(sigmoid(dot(v, u_pos)))
       -sum(log(sigmoid(-dot(v, u_neg))))
```

`logsigmoid` avoids numerically unstable manual sigmoid-then-log operations. Output embeddings start at zero; input embeddings start at small random values. A zero padding row is never sampled as a real word.

The objective makes observed word/context pairs more compatible than sampled noise. Negative sampling is not the classifier's non-scam class: the two uses of “negative” are unrelated.

I capped the initial experiment at 300,000 sampled pairs and reused them for three epochs. The sampler draws centers across the corpus, so it does not simply stop after reading the first few rows. Common words still get more chances because they occupy more positions; I have not implemented frequent-word subsampling. This keeps the experiment manageable, but it also limits what I can conclude about Word2Vec. A weak result from this run is not proof that every larger Word2Vec setup would fail.

Classification: average available token embeddings into one vector per complaint, standardize each dimension using training statistics, then fit a class-balanced logistic regression head. The embedding table is frozen for that head. Unseen Word2Vec tokens are skipped. An all-OOV document becomes a zero vector and receives a review flag.

Weaknesses: one vector per spelling, no native OOV representation, limited pretraining data, and mean pooling loses order and negation relationships. `send OTP` and `do not send OTP` can have overly similar pooled representations. A decreasing SGNS loss does not prove improved classification.

## 9. Approach C: character-subword Skip-gram extension

This extends the same training objective rather than claiming to reimplement every detail of fastText.

For each word, add boundary markers and generate character n-grams of length 3–5. Hash them into 50,000 buckets using stable BLAKE2b hashing. Store at most 24 unique n-gram IDs per word. Represent a center word as:

```text
center_vector(word) = learned_word_vector(word)
                      + mean(learned_ngram_vectors(word))
```

The shared character vectors receive gradients through the SGNS objective. Related spellings may share several n-grams, so they can share learned information. An unseen word uses its character vectors even without a learned whole-word vector.

Tradeoffs:

- Similar spelling does not guarantee similar meaning.
- Very different transliterations can share few useful n-grams.
- Hash collisions force unrelated n-grams to share parameters.
- Truncating to 24 n-grams bounds memory but discards some character information.
- Previously unused buckets are zeroed at export so unseen words are not represented using purely random, untrained bucket vectors.
- Placeholders do not get character n-grams, preventing number/PII replacement tokens from masquerading as spelling variants.
- The classifier still averages token vectors and lacks contextual understanding.

Training uses CUDA if available and falls back to CPU. The RTX 4060's 8 GB is ample for these compact embedding tables. The training loop limits CPU tensor threads and uses batches of 2,048 sampled pairs. Actual CUDA peak allocated memory is recorded per run; it is not total system-wide VRAM use.

## 10. Approaches considered and their status

| Approach | Status | Reason |
|---|---|---|
| Global English dictionary autocorrection | Rejected before training | Would change legitimate Hinglish, identifiers and names |
| Exhaustive Hinglish spelling dictionary | Rejected as a dependency | No complete canonical spelling; maintenance and ambiguity |
| Aggressive stemming/stopword deletion | Not used | Can lose negation and useful context; derived CSVs risk duplication |
| Token-level language ID followed by correction | Considered, deferred | Ambiguous short tokens; adds an error-prone stage before classification |
| Word TF-IDF | Trained and measured | Simple lexical baseline |
| Character TF-IDF | Trained and measured | Direct spelling robustness without mandatory correction |
| Word + character TF-IDF | Trained and measured | Combines lexical and subword evidence |
| Custom Word2Vec SGNS | Trained and measured | Educational baseline and lightweight representation experiment |
| Custom character-subword SGNS | Trained and measured | Tests spelling sharing and OOV representations |
| HingRoBERTa | Considered, not trained | Dedicated code-mixed candidate for a later comparison |
| Multilingual DistilBERT | Additional comparison implemented | Sparse/custom embedding validation was weak on complaints; measured results exist only if its JSON report is present |
| RAG/LLM report generation | Deferred by user scope | Retrieval cannot repair classifier label conflicts |

Do not say “transformers failed” based on a download or environment error. Check whether training and evaluation actually produced a report. Do not say subwords outperformed words unless the task-specific report demonstrates it. The goal of a benchmark is to learn, including when a simpler model wins.

## 11. Execution challenges and changes of direction

1. The default `python.exe` pointed to a WindowsApps alias and did not start. I used the existing NLP conda environment, which already had CUDA-enabled PyTorch, rather than changing the global installation.
2. The GPU was visible through `nvidia-smi`; PyTorch also confirmed CUDA availability. Driver-reported CUDA capability is not the same thing as the installed PyTorch wheel's CUDA runtime; record both rather than assuming they must match.
3. I found extensive duplication, so I deduplicated before splitting and removed complaint training templates that overlapped the supplied test set.
4. My first conflict-handling rule removed all inconsistent Hinglish groups. That also removed five scam labels. I changed this to an explicit `other_scam` mapping and recorded exactly which labels were merged.
5. The first sparse trial used SAGA logistic regression. I stopped it when the paired test file became available and rebuilt the evaluation split. I then used SGD logistic loss to keep the sparse experiments practical. Since SAGA did not finish that comparison, I cannot call it an accuracy failure.
6. The initial embedding-classifier trial hit an LBFGS iteration limit at 400 iterations. The run still returned predictions, but convergence was incomplete. The final configuration allows 1,000 iterations with a stated tolerance. If another warning occurs, record it rather than calling the solution fully converged.
7. Scikit-learn warned that `n_jobs` no longer affected LogisticRegression in my installed version, so I removed the obsolete argument.
8. The official `test.csv` was supplied during work. Complaint data preparation and model training were rerun; initial internal-test reports are archived as superseded.
9. Subword pooling repeats many token lookups. A bounded per-call cache reuses token representations without changing results; unused hash buckets are explicitly zeroed on export.
10. An early integration-test run started before complaint baseline training had saved its model. Preprocessing, grouping and gradient tests passed, but inference raised `FileNotFoundError` for the missing artifact. This was an execution-order problem, not a passing end-to-end check; the final test run is performed after all artifacts exist.
11. A source-inspection command hit Windows CP1252 encoding limits on the rupee sign. Data stayed intact; the problem was console encoding. CLI output is explicitly UTF-8 so English, Hinglish, Hindi and currency symbols can be displayed without that failure.
12. Three labels appear only in the complaint test file. I initially included them in the validation reporting vocabulary too, which lowered validation macro-F1 by adding zero-support classes. I corrected validation to use the training label vocabulary and retained unsupported labels in test metrics. The correction uses saved confusion counts, so predictions do not change. A regression test checks this boundary.
13. Exact duplicate checks were not sufficient for the call data. The family-held-out evaluation exposed the gap between recognizing familiar scripts and generalizing to a new script. This changed the headline result, not just a footnote.
14. The financial-expert trial hit a Windows/joblib shared-array error (`WRITEBACKIFCOPY base is read-only`) when the one-versus-rest heads ran in separate workers. I switched those heads to one worker. This changed the execution path, not the intended classifier objective.

The complaint failure analysis also explains why more model capacity did not automatically help. The baseline gets roughly 70% parent-category accuracy, but several attack subtypes have almost indistinguishable generic narratives. Their parent is easy to recognize while the specific subtype is close to guessing. I should not claim the classifier can infer SQL injection or ransomware from a story that gives no distinguishing technical evidence. The three test-only categories are another limit: no supervised model trained on the current label set can output those new labels correctly.

## 12. How to interpret evaluation

For a class, precision asks how many predicted examples were correct; recall asks how many actual examples were found. F1 is their harmonic mean. Macro-F1 averages class F1 values equally, so rare classes matter. Weighted-F1 weights class F1 by support; accuracy measures overall correct predictions. A majority-class baseline can have high accuracy and poor macro-F1.

Each report includes clean validation, noisy validation, clean test, and noisy test metrics, per-class scores and a confusion matrix. Noise swaps adjacent internal letters in 25% of eligible alphabetic words of length at least five; placeholders and numeric values are preserved. This is a synthetic stress test, not a natural Hinglish typo benchmark. Real transliteration changes, phonetic substitutions, ASR errors and adversarial obfuscation require additional tests.

I left duplicates and contradictory labels in the supplied test set. I also report a version deduplicated by `(template, label)`, so I can see how much repetition changes the result. Neither version removes disagreements just to improve the score. My grouping handles normalized template matches, not every possible paraphrase. A better benchmark would also separate template families, incidents and time periods.

The manually authored challenge includes English transfer, informal Hinglish, benign conversations, security warnings and negation. It is tiny and diagnostic. It is never added to fitting or used for model selection. Inspect failures, particularly high-confidence ones. If you start iterating against this challenge, it becomes development data and you need a fresh challenge set.

Review threshold: the default flag is set when the top score is below 0.55 or the representation has no known features. That value is a heuristic, not a validated operating point. A high score does not prove that an unfamiliar or benign input is in scope. Future work should calibrate probabilities, plot precision/recall at thresholds and add real out-of-domain examples.

I also added a class-level evidence check. The app flags a subtype if its evaluation F1 is below 0.5 or support is below 20. For the default call baseline it uses the harder family-CV summary; other models use validation. These are conservative heuristics, not calibrated confidence. For complaints, weak subtype evidence or disagreement between the top subtype and aggregated parent causes the suggested type to fall back to the broad category with a review note. The original ranked alternatives remain visible.

## 13. Why the current approach can be valuable

What I have now is a local pipeline I can inspect: the same preprocessing at training and inference, recorded label changes, fixed splits, saved models and comparisons that include weak results. Character features let me test spelling tolerance without rewriting the message. The custom SGNS models helped me understand embeddings, even where they did not perform well. The actual advantage of each model depends on the task-specific numbers below.

The part I need to explain in an interview is why the call split can score perfectly while new messages still fail. Saying “I got 100% accuracy” would leave out the most important finding: the dataset is small and repetitive, and the labels have problems.

## 14. Files to explain in a code review

| File | Responsibility |
|---|---|
| `cyberlens/preprocessing.py` | normalization, partial PII masking, grouping fingerprint, synthetic typo generation |
| `cyberlens/data.py` | task labels, conflicts, official-test overlap removal, deduplication and fixed splits |
| `cyberlens/baseline.py` | three sparse variants, validation-only selection, artifact saving |
| `cyberlens/embeddings.py` | vocabulary, pair sampler, SGNS, subwords, pooling and classifier head |
| `cyberlens/metrics.py` | aggregate/per-class metrics and synthetic-noise evaluation |
| `cyberlens/predict.py` | trusted artifact loading, consistent inference, task scope and review flag |
| `cyberlens/desktop.py` | local Tkinter interface with background inference |
| `cyberlens/challenge.py` | small transparent hand-authored diagnostic |
| `cyberlens/report.py` | measured results, environment and source-integrity report |
| `tests/test_pipeline.py` | meaningful invariants: no split leakage, masking, gradient flow and inference |

## 15. Interview questions and defensible answers

**Why not correct every misspelling first?** Romanized Hindi has variable spellings and ambiguous tokens. A global correction rule can change the meaning. I tested spelling-tolerant representations first and left uncertain words alone.

**Why is Word2Vec not enough?** It learns one vector per vocabulary spelling, has no native unseen-word representation, and is not contextual. Our pooled classifier also loses word order.

**How does your custom subword model help?** It shares trained character n-gram vectors across spellings and can construct a vector for an unseen spelling. That is a mechanism, not a guarantee of better downstream accuracy.

**Did you implement fastText?** I implemented a fastText-inspired subword extension to SGNS. I did not reproduce the full library or all its training optimizations.

**Why didn't you use a transformer immediately?** The first blocker was conflicting labels and duplicate leakage. A larger model would not solve those. After establishing baselines, weak complaint validation motivated an additional multilingual encoder comparison. Its actual status and measured outcome are in the reports.

**How did you prevent leakage?** I fitted vocabularies and embeddings on training data only, separated normalized templates, removed complaint training overlaps with the supplied test set, and selected models using validation. Paraphrase-level leakage is still a limitation I need to test more carefully.

**Are outputs safe/unsafe verdicts?** No. The complaint head only categorizes incidents and has no benign class. Call labels have a benign class but narrow training coverage. Scores are uncalibrated and require review.

**Can it detect all seven scam subtypes?** Not from the supplied contradictory call labels. It supports four coherent call labels and a separate broader complaint taxonomy. Fine subtypes require better labeled evidence.

**Does masking remove all PII?** No. It masks selected structured candidates. Name/address NER, stronger validation and entity-level recall auditing are required before sharing.

**Where is GenAI?** SGNS is representation learning and the deployed heads perform classification. This version is not a generative reporting system. A later grounded report-drafting layer can satisfy that separate requirement.

**What would you improve next?** Correct/align labels across languages, collect independent benign and scam messages, evaluate template-family/time-based splits, improve privacy, compare contextual encoders, calibrate confidence, and only then add grounded guidance.

## 16. RAG upgrade boundary

Keep classification evidence separate from retrieved guidance. A later RAG component can retrieve reviewed incident-response material using the message, predicted alternatives and extracted evidence. Retrieval should not turn an uncertain prediction into a fact. Report generation must avoid invented incident details and should explicitly mark missing facts. Keep user review before sharing anything externally.

Before that upgrade, the largest improvement opportunity is aligned, diverse, correctly labeled English/Hinglish data—not a more elaborate prompt or automatic spelling dictionary.

## 17. Primary references

- [Original Word2Vec architectures](https://arxiv.org/abs/1301.3781)
- [Negative sampling and related extensions](https://arxiv.org/abs/1310.4546)
- [fastText word/subword representation tutorial](https://fasttext.cc/docs/en/unsupervised-tutorial.html)
- [Scikit-learn SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- [HingRoBERTa model card](https://huggingface.co/l3cube-pune/hing-roberta)
- [Presidio masking architecture](https://microsoft.github.io/presidio/text_anonymization/)

These explain the referenced mechanisms; they do not establish CyberLens performance. Only the locally generated reports describe this implementation's measured outcomes.

## 18. Additional optimizer checks and contextual encoder mechanism

Three extra word-feature checks use the training/validation splits only: SGD with stronger regularization, SGD without inverse-frequency class weighting, and balanced LinearSVC. `reports/optimizer_validation.json` records their actual metrics. None materially improved the initial complaint validation macro-F1. That supports investigating representations and data rather than blindly changing only the optimizer. No untested SVM probability claims are made.

The additional encoder is `distilbert/distilbert-base-multilingual-cased`: a public pretrained multilingual encoder with a newly initialized task-specific sequence-classification head. This is **fine-tuning**, not a transformer implemented or pretrained from scratch. Downloading weights does not upload complaint text. Model files are cached inside this folder. Network restrictions initially caused DNS failures; that is an environment failure, not a model accuracy result.

Its subword tokenizer produces token IDs and attention masks. The six-layer encoder computes contextual token representations using self-attention; the sequence classification head produces one logit per task label. Unlike static Word2Vec, a token representation can depend on surrounding words. This can help context and negation, but is not guaranteed to understand Romanized Hindi or unseen scam tactics.

For complaint output, the UI also sums subtype scores belonging to the same parent category. That gives a broad-category alternative, which can differ from the parent of the single most likely subtype. This is score aggregation, not a separately trained hierarchical classifier or a calibrated broad-category guarantee.

The 8 GB training profile uses 192-token truncation, microbatches of 8, accumulation across four microbatches (effective batch size approximately 32), bfloat16 mixed precision and gradient checkpointing. Accumulation reduces per-step activation memory at the cost of more microsteps. Checkpointing recomputes activations during backward rather than storing all of them. BF16 has a wider exponent range than FP16 and does not use FP16 gradient scaling in this implementation. AdamW updates the pretrained encoder and new head; a 10% warmup precedes linear learning-rate decay from 3e-5. Gradients are clipped at norm 1.

After profiling the small-batch run, I restarted complaint training with a microbatch of 32 and accumulation of 1. That keeps the effective batch size at 32 and uses the GPU more efficiently. The call encoder uses the original 8-by-4 setup. Both configurations and their actual memory use are recorded in the model reports; the small-batch trial log is kept locally.

Thirty percent of training examples receive deterministic synthetic typo augmentation at a 12% eligible-word rate; test text is never used to create training variants. Cross-entropy uses square-root inverse-frequency weights, a gentler reweighting than the sparse model's full inverse frequency. The epoch budget is fixed, and validation compares models after training; no checkpoint is selected on test scores. Long-message truncation can discard crucial late evidence and is explicitly a limitation.

If the contextual model completes, it is eligible for `best` backend selection using the same clean/noisy validation macro-F1 rule. Otherwise the completed local baselines remain usable and no transformer performance is invented. The model card is [here](https://huggingface.co/distilbert/distilbert-base-multilingual-cased).

I also tried a smaller, targeted change: a dedicated financial-fraud head. It learns the seven financial subtypes using training complaints from that parent category. At inference I keep the base model's total probability mass for financial fraud and redistribute it using the specialist's conditional probabilities. Other categories keep their original mass. Three regularization/weighting choices are compared using the full validation taxonomy, not just financial examples. The test report uses the same full complaint taxonomy as the other models, so this is not an accuracy gain created by silently dropping hard classes. The app keeps this experiment only if it wins the existing validation rule.

That experiment did not win. Its best clean validation macro-F1 was about 0.264, but the typo result was lower, leaving the combined selection score below the existing baseline. Its supplied-test macro-F1 was about 0.231 versus 0.234 for the baseline. I kept the simpler model as the default. That is the outcome I should explain, rather than calling the extra head an improvement just because it adds another model.

The initial Hugging Face helper fetched small files but stalled on the large weight file. A separate HTTPS probe confirmed the public model file was reachable. The project includes `download_encoder.py`, which downloads only the public assets into the new folder, pins the publisher's revision and validates the weight file against the publisher's LFS SHA-256. Fine-tuning then loads that local directory. The fallback is an infrastructure workaround, not a change of model or an accuracy comparison.

<!-- GENERATED RESULTS -->

## 19. Final measured outcomes (automatically generated)

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
