# Fine-Grained Emotion Classification in Mobile App Reviews

Reproduction of published human and large language model (LLM) evaluations, followed by supervised machine learning experiments for multi-label emotion classification.

This project builds on *What About Emotions? Guiding Fine-Grained Emotion Extraction from Mobile App Reviews*. It investigates eight emotions in app-review sentences, compares classical classifiers, and evaluates class weighting and emotion-specific thresholds. An interactive demonstration accepts a new review and returns predicted emotions.

**Contributors:** Shraddha Debata, Sreya, and Jaspreet.

**Current status:** Experiments 1–5 are implemented, and Experiment 6 provides interactive inference. Threshold tuning improves validation F1 but reduces test F1. Training diagnostics also show a substantial generalization gap for the SVM models.

> **Results provenance:** The primary results below come from the saved outputs in `Emotion_Classification_Project_3.ipynb`, reviewed on November 24, 2026. They supersede figures from earlier project snapshots. A separate local diagnostic rerun is explicitly identified and is not substituted for the notebook's recorded results.

## Contents

- [Project objectives](#project-objectives)
- [Experiments and status](#experiments-and-status)
- [Dataset](#dataset)
- [Evaluation protocol](#evaluation-protocol)
- [Experimental methods](#experimental-methods)
- [Results](#results)
- [Overfitting assessment](#overfitting-assessment)
- [Running the project](#running-the-project)
- [Interactive review classification](#interactive-review-classification)
- [Files and outputs](#files-and-outputs)
- [Verification and reproducibility](#verification-and-reproducibility)
- [Limitations and remaining work](#limitations-and-remaining-work)
- [References and attribution](#references-and-attribution)

## Project Objectives

1. Reproduce the released dataset statistics and document differences from the paper.
2. Recalculate human agreement and LLM performance from the authors' saved annotations.
3. Compare a training-prevalence baseline, Logistic Regression, and Linear SVM.
4. Investigate whether balanced class weights improve rare-emotion recognition.
5. Optimize emotion-specific thresholds on validation data and measure their effect on test performance.
6. Demonstrate predictions for a user-entered review without retraining.

The paper-reproduction track evaluates **ten annotation labels on all 1,112 records**. The supervised extension evaluates **eight emotion labels on a separate test partition**. These scores use different evaluation populations and should not be treated as a direct ML-versus-LLM comparison.

## Experiments and Status

| Experiment | Status | Main outputs |
| --- | --- | --- |
| **1. Dataset reproduction** | Implemented; saved outputs present | Dataset statistics, label frequencies, multi-label counts, duplicate analysis, and published-versus-released comparisons |
| **2. Published LLM metrics** | Implemented; saved outputs present | Tables 1–6, agreement calculations, proposal Figures 3–4, historical cost analysis, and comparison tables |
| **3. ML classification** | Implemented; saved outputs present | Bounded LR/SVM search, baseline comparison, per-emotion precision/recall/F1, timing, and model sizes |
| **4. Rare-emotion classification** | Implemented; saved outputs present | Unweighted-versus-balanced SVM comparison and validation-selected model |
| **5. Threshold optimization** | Implemented; saved outputs present | Frozen emotion-specific cutoffs, constraint status, default-versus-tuned metrics, and rare-label analysis |
| **6. Interactive classification** | Implemented; saved example present | Review input and predicted emotions; a small output-labeling issue is documented below |

Implementation does not imply that a proposed improvement succeeded. Experiment 5 is a completed experiment with a negative overall test-set result.

## Dataset

The released annotations describe mobile-app review sentences across ten Google Play categories.

| Property | Value |
| --- | ---: |
| Annotated records | 1,112 |
| Distinct sentence strings | 1,108 |
| Raw application names | 259 |
| Normalized application names | 257 |
| Positive labels across all ten annotation columns | 1,274 |
| Multi-label records across all ten columns | 157 |
| Multi-label records among the eight emotions | 153 |

### Label Frequencies

| Label | Paper | Released dataset | Difference |
| --- | ---: | ---: | ---: |
| Joy | 372 | 372 | 0 |
| Trust | 151 | 152 | +1 |
| Fear | 24 | 24 | 0 |
| Surprise | 54 | 54 | 0 |
| Sadness | 303 | 303 | 0 |
| Disgust | 49 | 50 | +1 |
| Anger | 27 | 27 | 0 |
| Anticipation | 182 | 182 | 0 |
| Neutral | 88 | 88 | 0 |
| Reject | 22 | 22 | 0 |
| **Total** | **1,272** | **1,274** | **+2** |

The released labels remain unchanged. Differences are reported rather than removed to force agreement with the paper.

### Supervised Targets

```text
Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation
```

The supervised track excludes all 22 Reject-marked records while preserving their original partition membership. It retains 86 Neutral-only records as eight-zero targets and keeps emotion labels on the two mixed Neutral-plus-emotion records. Neutral and Reject are not separate supervised outputs.

An empty set of predicted emotions means **“No predicted emotion,” not confirmed Neutral**.

## Evaluation Protocol

### Partitions and Leakage Checks

Records sharing a review ID or normalized sentence are joined into connected groups. Grouped splitting with seed 42 creates the train, validation, and test partitions. Reject exclusions are applied afterward.

| Partition | Original records | Eligible supervised records |
| --- | ---: | ---: |
| Training | 775 | 761 |
| Validation | 169 | 166 |
| Test | 168 | 163 |
| **Total** | **1,112** | **1,090** |

The later Experiment 3 preparation code checks record identities, review/text separation, and connected-group separation. It also screens long texts for conservative lexical near duplicates using three-word-shingle Jaccard similarity of at least 0.95.

TF-IDF and classifiers are fitted on training data only. Validation selects configurations, class weighting, and thresholds. Test labels are used for reporting after the corresponding choices are frozen.

The earlier exploratory ten-label models use a different random 80/20 split. Their scores are historical exploration, not the supervised benchmark reported here. Creating a grouped split later does not retroactively change those earlier models.

### Metrics

The supervised experiments use standard precision, recall, and F1 for each emotion, alongside micro-F1, macro-F1, rare-emotion macro-F1, and Hamming loss where reported. Undefined precision/recall is reported as zero through `zero_division=0`.

- **Micro-F1:** aggregates label-level decisions across records and emotions.
- **Macro-F1:** averages the eight emotion-specific F1 scores equally.
- **Rare-emotion macro-F1:** averages Fear, Anger, Disgust, and Surprise.
- **Hamming loss:** fraction of individual label decisions that are incorrect; lower is better.

The paper-reproduction code retains inferred paper-specific aggregation conventions separately. In particular, some human-table “F1” values are reproduced by an arithmetic mean of precision and recall. That is not standard F1; the new supervised experiments use standard F1 throughout.

## Experimental Methods

### Experiment 1: Dataset Reproduction

Validate record counts, missing sentences, application/category counts, label totals, duplicate texts, and multi-label examples. Compare measured values with the paper and retain documented discrepancies.

### Experiment 2: Published Human and LLM Evaluations

Load the authors' saved annotations for GPT-4o, Mistral Large, and Gemini 2.0 Flash, using three temperature-zero runs per model.

Combine predictions with two-stage majority voting:

1. Retain each label supported by at least two of a model's three runs.
2. Retain each label supported by at least two of the three model consensuses.

This differs from a single vote across nine runs. Preserve records without a combined consensus label. Recalculate correctness and human/human–LLM Cohen's kappa from aligned records, including shared-iteration agreement summaries.

The notebook generates Tables 1–6 and the proposal's Figures 3 and 4. Higher-temperature files are inventoried and validated separately. Historical time/cost calculations use the released workbook, not current API prices or new provider calls.

### Experiment 3: Machine Learning Classification

Compare a training-prevalence baseline with TF-IDF Logistic Regression and Linear SVM using eight one-versus-rest classifiers. Always-negative and most-frequent-label predictors are additional diagnostics.

| Search parameter | Values |
| --- | --- |
| Algorithm | Logistic Regression, Linear SVM |
| TF-IDF n-grams | Unigrams; unigrams and bigrams |
| C | 0.5, 1.0, 2.0 |
| Maximum vocabulary | 5,000 features |
| Term weighting | Sublinear TF |
| Accent handling | Unicode accent stripping |
| Stop-word removal | None |
| Class weights | None |
| Seed | 42 |

The fixed search evaluates 12 configurations. Selection maximizes validation macro-F1, then micro-F1; ties prefer unigrams and smaller C. The selected LR uses unigrams and C=2. The selected SVM uses unigrams/bigrams and C=2 and is the overall validation winner.

### Experiment 4: Balanced Class Weights

Reuse the Experiment 3 winner and change only `class_weight=None` to `class_weight="balanced"`. Keep the algorithm, feature parameters, C, partitions, and default decision boundaries fixed.

For each emotion's binary classifier:

```text
Positive weight = training rows / (2 × positive training examples)
Negative weight = training rows / (2 × negative training examples)
```

The balanced SVM is selected for Experiment 5 using validation macro-F1, with micro-F1 as the tie-breaker. The original unweighted pipeline remains available.

### Experiment 5: Emotion-Specific Thresholds

Use the selected fitted model without refitting. Search 0.10–0.90 in steps of 0.05 for each emotion, including the original decision boundary as an additional candidate.

For Logistic Regression, these are literal probability thresholds. The executed experiment uses Linear SVM, whose margins are not probabilities. For SVM, the grid specifies positions across each emotion's validation margin range:

```python
raw_cutoff = validation_min + grid_value * (validation_max - validation_min)
```

This mapping is a fraction of the score range, not a probability or percentile. The original SVM cutoff is zero. Predictions use a strict `score > cutoff` comparison.

Select maximum validation F1 subject to precision of at least 40% and at least one positive prediction. Break ties by precision, preference for the default, and proximity to the default. Freeze the resulting raw cutoffs before test evaluation; never rescale them on a new review or test batch.

Fear has no feasible candidate under this grid and constraint, so the tuned policy abstains and records the failure. The validation precision floor is not a test-set guarantee. The helper supports LR thresholds, but the saved experiment does not report a separately tuned LR model.

### Experiment 6: Interactive Review Classification

The input interface accepts a review and displays one or more predicted emotions. The default interface uses the Experiment 4 balanced model; the Experiment 5 cutoff artifact is optional. This is an inference demonstration, not a new accuracy experiment or model-selection result.

Prefer one sentence at a time because the evaluated dataset uses sentence-level examples. The interface does not provide calibrated confidence percentages, detect Reject, or establish a Neutral label.

## Results

All supervised values in this section are the **attached notebook's saved outputs**, rounded to four decimal places. All approaches use the same 163 eligible test records.

### Paper Reproduction

| Metric | Published target | Recomputed |
| --- | ---: | ---: |
| Final human-pair agreement | Approximately 0.69 | 0.694 |
| Combined LLM precision | 70.68% | 70.68% |
| Combined LLM recall | 66.81% | 66.80% |
| Combined LLM paper-compatible F1 | 68.69% | 68.68% |

The notebook retains published-versus-measured comparisons and distinguishes paper-compatible aggregate scores from standard F1.

### Experiment 3: Held-Out Comparison

| Approach | Micro precision | Micro recall | Micro-F1 | Macro-F1 | Rare macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prevalence baseline | 0.2362 | 0.8508 | 0.3697 | 0.1865 | 0.0000 |
| Logistic Regression | 0.8226 | 0.2818 | 0.4198 | 0.1907 | 0.0000 |
| Linear SVM | 0.7250 | 0.4807 | 0.5781 | 0.3287 | 0.0417 |

The notebook also records fit time, prediction time, and serialized model size. These measurements depend on the runtime and do not represent total search cost, peak RAM, or deployment cost.

### Experiment 4: Class Weighting

| Metric | Unweighted SVM | Balanced SVM |
| --- | ---: | ---: |
| Validation macro-F1 | 0.3511 | 0.3654 |
| Test micro precision | 0.7250 | 0.6741 |
| Test micro recall | 0.4807 | 0.5028 |
| Test micro-F1 | 0.5781 | 0.5759 |
| Test macro-F1 | 0.3287 | 0.3434 |
| Test rare-emotion macro-F1 | 0.0417 | 0.0742 |

Weighting improves validation and test macro-F1 and test rare-emotion macro-F1. Test micro-F1 decreases slightly, so the improvement is not uniform across metrics.

### Experiment 5: Threshold Tuning

| Metric | Default balanced SVM | Tuned cutoffs |
| --- | ---: | ---: |
| Validation micro-F1 | 0.5629 | 0.6171 |
| Validation macro-F1 | 0.3654 | 0.4450 |
| Validation rare-emotion macro-F1 | 0.1548 | 0.2549 |
| Test micro precision | 0.6741 | 0.5604 |
| Test micro recall | 0.5028 | 0.5635 |
| Test micro-F1 | 0.5759 | 0.5620 |
| Test macro-F1 | 0.3434 | 0.3277 |
| Test rare-emotion macro-F1 | 0.0742 | 0.0662 |

Tuning increases test recall but reduces overall precision and F1. Validation macro-F1 increases by 0.0796 while test macro-F1 decreases by 0.0157.

| Rare emotion | Positive test examples | Default test F1 | Tuned test F1 |
| --- | ---: | ---: | ---: |
| Fear | 2 | 0.0000 | 0.0000 |
| Anger | 4 | 0.0000 | 0.0000 |
| Disgust | 11 | 0.1538 | 0.1538 |
| Surprise | 10 | 0.1429 | 0.1111 |

No rare emotion improves its test F1. Sadness improves from 0.5859 to 0.6042, while Joy and Trust decline. Report these mixed results rather than claiming that threshold tuning improves every emotion.

## Overfitting Assessment

### Findings from the Attached Notebook

Experiment 5 shows a pattern **consistent with validation overfitting**: the optimization improves validation F1 substantially but reduces held-out test F1. With only 166 validation records and very few rare positives, chosen cutoffs can adapt to individual validation examples.

This is evidence of poor generalization of the chosen thresholds, not proof of a single cause or a statistical significance result. Class imbalance, partition differences, and sampling variation also affect the scores.

Experiment 4's macro-F1 improvement appears in both validation and test, so class weighting generalizes more consistently on this split. Experiment 6 does not fit anything and therefore introduces no additional training overfitting; it inherits the selected model's limitations.

### Independent Training Diagnostic

The attached notebook does not report a complete train/validation/test performance table for the selected models. A separate audit fitted the same three selected configurations on the exact uploaded dataset and matching grouped partition, using the pinned local environment. It did not tune configurations or overwrite the notebook's models.

| Model | Training macro-F1 | Validation macro-F1 | Test macro-F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.3451 | 0.2094 | 0.1907 |
| Unweighted SVM | 1.0000 | 0.3475 | 0.2990 |
| Balanced SVM | 1.0000 | 0.3862 | 0.3408 |

Perfect training F1 combined with substantially lower unseen-data F1 is a strong training-overfitting signal for the SVMs in this rerun. LR also has a generalization gap, alongside low overall performance.

**These diagnostic rerun figures differ from the saved notebook's SVM outputs.** The cause was not established. Runtime/library differences or notebook execution history require investigation; neither is asserted as the confirmed explanation. Do not combine results from the two runs into one model comparison.

See [the overfitting review](OVERFITTING_REVIEW.md), [diagnostic metrics](overfitting_audit/local_training_validation_test.csv), and [saved notebook evidence](overfitting_audit/notebook_evidence.json).

### Next Evaluation Steps

- Add train/validation/test diagnostics in the same Colab session to confirm the current fitted models' training gaps.
- Compare stronger regularization and smaller feature spaces using grouped cross-validation within development data.
- Where feasible, assess threshold stability with out-of-fold predictions and resampling that respects review/text groups and rare-label support.
- Freeze the protocol before further testing. Repeated decisions based on the same observed test results weaken its role as an independent evaluation.
- Report uncertainty and support counts; one or two rare examples can materially change F1.

These are proposed next steps, not completed experiments.

## Running the Project

### Google Colab

1. Open `Emotion_Classification_Project_3.ipynb` in Colab.
2. Run the notebook in order, including the grouped split and Experiments 3–6.
3. Upload `ground-truth.xlsx` when prompted.
4. Provide `emotion_replication_bundle.zip` when the paper-reproduction setup requests it. It must contain the replication scripts and original human/LLM annotation resources.
5. Review generated tables and plots, then download any artifacts you want to preserve.

A CPU runtime is sufficient. Reproducing the saved LLM evaluations and using the interactive classifier requires no new provider API calls.

### Local Environment

The notebook includes Colab-specific upload/mount cells. For local execution, replace those setup steps with local data paths; installing dependencies alone does not make those Colab cells portable.

The existing local artifact environment uses Python 3.12 and the versions in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Direct requirements include NumPy, SciPy, pandas, openpyxl, scikit-learn, matplotlib, and the notebook execution tools. `joblib` is installed through scikit-learn and is used for artifacts.

The pinned environment supports the separate local audit. The exact package versions of the attached Colab run are not recorded in its saved outputs; exact numerical agreement is therefore not established.

## Interactive Review Classification

Run the Experiment 6 helper and model-setup cells first. The setup can use the current `best_pipeline_after_exp4` variable or its saved model file.

```python
EXP6_MODE = "default"  # Or "tuned" for Experiment 5's fixed cutoffs.
```

After changing the mode, rerun the setup cell, then:

```python
exp6_result = enter_review(exp6_model, mode=EXP6_MODE)
```

The cell asks:

```text
Enter your review:
```

Enter a review and press Enter. The model prints the predicted emotions. Rerun the cell for another review. Alternatively:

```python
result = classify_review(
    "I love the app, but the latest update keeps crashing.",
    exp6_model,
    mode=EXP6_MODE,
)
print(result["classification"])
```

After a runtime restart, rerun the model-producing cells or upload the saved artifact and set its path. For tuned inference, use the helper that applies the saved raw cutoffs: the contained pipeline's ordinary `.predict()` ignores the tuned cutoffs.

**Current display issue:** The attached final cell predicts a fixed example, asks for user input, then prints the fixed example's classification under “Predicted overall emotion(s).” That final line can appear to describe the entered review. Use `exp6_result["classification"]` for the entered review, or keep only the `enter_review(...)` call above. This review documents the issue without changing the original notebook.

## Files and Outputs

The notebook is the primary implementation. The following layout summarizes the notebook, review files, and runtime resources; it is not a claim that every historical resource is included in a single download.

```text
.
├── README.md
├── Emotion_Classification_Project_3.ipynb
├── OVERFITTING_REVIEW.md
├── Overfitting_Check_Ready_To_Paste.ipynb
├── requirements.txt
├── ground-truth.xlsx                         # Dataset input
├── emotion_replication/                      # Original replication resources
│   ├── replication.py
│   ├── supplementary.py
│   ├── data/
│   └── experiment_2/                         # Recomputed figures and tables
├── manual_continuation_results/              # Earlier reproduction tables
├── experiment_3_results/                     # Runtime-generated artifacts
├── experiment_4_results/
├── experiment_5_results/
└── overfitting_audit/                        # Separate local review evidence
    ├── local_training_validation_test.csv
    └── notebook_evidence.json
```

Important runtime artifacts:

| File | Purpose |
| --- | --- |
| `experiment_3_results/split_manifest.csv` | Exact eligible partition identities and order |
| `experiment_3_results/validation_search.csv` | All 12 configuration scores |
| `experiment_3_results/frozen_selection.json` | Validation-only configuration selection |
| `experiment_3_results/best_pipeline.joblib` | Selected unweighted pipeline |
| `experiment_4_results/comparison_all_emotions.csv` | Per-emotion class-weighting comparison |
| `experiment_4_results/best_after_experiment4.joblib` | Validation-selected pipeline used by the next experiment |
| `experiment_5_results/validation_thresholds.csv` | Cutoffs, validation metrics, and constraint status |
| `experiment_5_results/validation_threshold_search.csv` | Evaluated threshold candidates |
| `experiment_5_results/frozen_thresholds.json` | Frozen raw cutoffs and provenance |
| `experiment_5_results/comparison_all_emotions.csv` | Default/tuned test precision, recall, and F1 |
| `experiment_5_results/comparison_rare_emotions.csv` | Rare-emotion effects |
| `experiment_5_results/threshold_optimized_model.joblib` | Dictionary containing the pipeline, cutoffs, and metadata |

Local `experiment_3_results`–`experiment_5_results` files from previous completion bundles represent earlier runs. They must not be presented as the artifacts that produced the newly attached notebook's different SVM scores. Regenerate them in one clean run before packaging final results.

## Verification and Reproducibility

This review inspected all 163 cells and their saved outputs. No saved error outputs were found. This does not establish that a fresh “Run all” completes: original Colab uploads, external resources, and interactive input were not rerun during this review.

The audit verified that the uploaded dataset embedded in the notebook exactly matches the local diagnostic dataset and that recreated supervised identities match the prior split manifest. It fitted three fixed diagnostic models and left the original notebook unchanged.

The companion diagnostic notebook checks current model performance without fitting or selecting anything. Add it after Experiment 5 to obtain training gaps for the current Colab model objects.

For a final reproducible release, record Python/package versions, execute from a clean runtime, export current CSVs and models together, and preserve the dataset hash, split manifest, selection files, and threshold metadata.

## Limitations and Remaining Work

- The dataset is small and imbalanced; test support is only two Fear and four Anger examples.
- Strong SVM training fit does not translate into comparable unseen-data F1.
- Validation-selected thresholds do not improve overall test F1 in this run.
- Passing exact-overlap checks does not prove the absence of every semantic duplicate.
- The split groups reviews and repeated sentences, not entire applications. These results do not establish performance on wholly unseen apps.
- Released LLM annotations are historical outputs, not new blind model generations.
- Model inference can miss emotions, assign incorrect emotions, or return no label.
- Current runtime artifacts and saved notebook scores need consolidation into one clean reproducible run.
- The final same-test, eight-label ML-versus-LLM comparison, grouped cross-validation, uncertainty analysis, and final research report remain pending.

## References and Attribution

Quim Motger, Marc Oriol, Max Tiessler, Xavier Franch, and Jordi Marco. *What About Emotions? Guiding Fine-Grained Emotion Extraction from Mobile App Reviews.* 2025 IEEE 33rd International Requirements Engineering Conference (RE), pp. 6–18.

- [Research paper](https://doi.org/10.1109/RE63999.2025.00012)
- [Released dataset and replication package](https://doi.org/10.6084/m9.figshare.28548638)

```bibtex
@inproceedings{motger2025whataboutemotions,
  author    = {Motger, Quim and Oriol, Marc and Tiessler, Max and
               Franch, Xavier and Marco, Jordi},
  title     = {What About Emotions? Guiding Fine-Grained Emotion
               Extraction from Mobile App Reviews},
  booktitle = {2025 IEEE 33rd International Requirements Engineering
               Conference (RE)},
  year      = {2025},
  pages     = {6--18},
  doi       = {10.1109/RE63999.2025.00012}
}
```

Original dataset, annotations, guidelines, and replication materials belong to their respective authors. The project adds reproduction analysis and supervised experiments. The local resource bundle includes the authors' `AUTHORS_LICENSE.md`; no separate license for newly authored project code is declared in this README.
