# Fine-Grained Emotion Classification in Mobile App Reviews

**Reproduction of published human/LLM evaluations and classical machine-learning extensions for eight-emotion classification.**

This project studies emotions expressed in mobile-app reviews. It reproduces the released results from [*What About Emotions? Guiding Fine-Grained Emotion Extraction from Mobile App Reviews*](https://doi.org/10.1109/RE63999.2025.00012), then evaluates TF-IDF-based Logistic Regression and Linear SVM models.

The current sequential implementation completes **Experiments 1–4**. **Experiment 5 remains pending.**

**Project contributors:** Shraddha Debata, Sreya Mathew, and Jaspreet Aujla.

## Table of Contents

- [Project Overview](#project-overview)
- [Current Status](#current-status)
- [Dataset](#dataset)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Outputs and Saved Models](#outputs-and-saved-models)
- [Validation](#validation)
- [Limitations](#limitations)
- [Remaining Work](#remaining-work)
- [References and Attribution](#references-and-attribution)

## Project Overview

Traditional sentiment analysis labels reviews as positive, negative, or neutral. This project examines more specific emotions to distinguish praise, disappointment, concern, rejection, and expectations about app features.

The project has two tracks:

1. **Paper reproduction:** use the authors' released human and LLM annotations to recompute dataset statistics, agreement scores, classification metrics, figures, and historical time/cost estimates.
2. **Supervised extension:** train and evaluate classical models on eight emotion labels, then investigate whether balanced class weights improve recognition of rare emotions.

> The reproduction track uses ten annotation labels and the full dataset. The supervised track uses eight emotion labels and a held-out test set. Their aggregate scores must not be compared directly as if they represented the same evaluation.

## Current Status

| Experiment | Status | Implemented work |
| --- | --- | --- |
| **1. Original dataset statistics** | ✅ Complete | Dataset size, normalized app counts, label frequencies, multi-label records, duplicates, and published-versus-released comparisons |
| **2. Published human/LLM metrics** | ✅ Complete using released annotations | Nine LLM runs, two-stage voting, human/LLM Cohen's kappa, reproduced figures, Tables 1–6, historical cost estimates, and comparison tables |
| **3. ML emotion classification** | ✅ Complete | Training-prevalence baseline, diagnostic baselines, eight-label LR/SVM models, a bounded validation search, per-emotion metrics, and computational-efficiency tables |
| **4. Rare-emotion improvement** | ✅ Complete | Controlled unweighted-versus-balanced comparison of the Experiment 3 winner, rare-label results, and validation-based selection for the next experiment |
| **5. Emotion-specific thresholds** |  Pending | Validation-only cutoff optimization with a precision constraint and default-versus-tuned test comparisons |

The canonical notebook for this stage is:

[`Emotion_Classification_Project_3_Experiments_3_4_Completed.ipynb`](Emotion_Classification_Project_3_Experiments_3_4_Completed.ipynb)

Earlier exploratory notebooks may use different splits or label sets. The supervised results reported below come from the current sequential Experiment 3/4 implementation.

## Dataset

The original dataset and replication resources are available through [Figshare](https://doi.org/10.6084/m9.figshare.28548638).

| Property | Released dataset |
| --- | ---: |
| Annotated records | 1,112 |
| Distinct sentence strings | 1,108 |
| Raw application names | 259 |
| Application names after Unicode, spacing, and case normalization | 257 |
| Google Play categories | 10 |
| Positive annotation labels across all ten columns | 1,274 |
| Records with multiple annotation labels across all ten columns | 157 |
| Records with multiple labels among the eight emotions only | 153 |

### Labels and published-count differences

| Label | Published count | Released count | Difference |
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

The released annotations are preserved. No labels are removed to force agreement with the paper's printed counts.

### Supervised data policy

The supervised target columns are:

```text
Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation
```

- Exclude all **22 Reject-marked records** from the supervised track, including two Reject+emotion conflicts. This is a documented conservative policy, not an annotation correction.
- Keep **86 Neutral-only records** as eight-zero emotion targets.
- Preserve emotion labels on the two mixed Neutral+emotion records.
- Preserve the existing grouped split membership when applying these exclusions.
- Keep the original 1,112-row, ten-label dataset for paper reproduction.

| Partition | Before supervised exclusions | Experiment 3/4 rows |
| --- | ---: | ---: |
| Training | 775 | 761 |
| Validation | 169 | 166 |
| Test | 168 | 163 |
| **Total** | **1,112** | **1,090** |

A prediction containing no emotion is **not proof of Neutral**. The supervised models do not train separate Neutral or Reject detectors.

## Experimental Design

### Experiment 1: Reproducing Dataset Statistics

Load the released ground truth and verify the number of records, application names, categories, annotation labels, repeated texts, and multi-label examples. Export comparisons with the paper's reported statistics and retain the observed discrepancies.

### Experiment 2: Reproducing Published LLM Metrics

Use the authors' saved temperature-zero predictions from:

- GPT-4o: three runs.
- Mistral Large: three runs.
- Gemini 2.0 Flash: three runs.

Rebuild agreement in two stages:

1. Keep a label when at least **two of three runs within a model** agree.
2. Keep a label when at least **two of the three model consensuses** agree.

Do not replace this hierarchy with a single vote across nine runs. Preserve the **39 records without a combined consensus label**.

Reproduction includes the human-agreement evolution chart and human/LLM agreement heatmap embedded as Figures 3 and 4 in the proposal. Pairwise kappa is calculated within shared annotation iterations and then averaged. Missing human pairs remain missing rather than being filled with zero agreement.

Metrics are recomputed from annotations. Printed paper values are comparison targets, not substitutes for measured results. No new LLM/API generations are needed.

### Experiment 3: Machine Learning Emotion Classification

Compare three main approaches:

1. **Prevalence baseline:** predict every emotion occurring in at least 10% of training records.
2. **TF-IDF + Logistic Regression:** eight one-versus-rest binary classifiers.
3. **TF-IDF + Linear SVM:** eight one-versus-rest binary classifiers.

Always-negative and most-frequent-label rules are additional diagnostics.

The fixed search contains **12 configurations**:

| Setting | Values |
| --- | --- |
| Algorithm | Logistic Regression, Linear SVM |
| TF-IDF n-grams | Unigrams; unigrams and bigrams |
| C | 0.5, 1.0, 2.0 |
| Maximum features | 5,000 |
| Class weights | None |
| Stop-word removal | None; negations remain available |
| Random seed | 42 |

Fit TF-IDF only on training text. Select configurations using **validation macro F1**, then micro F1. Ties prefer unigrams and smaller C. Freeze the choices before scoring test data.

Validation selected:

- **LR:** unigrams, C=2.
- **SVM:** unigrams and bigrams, C=2.
- **Overall ML winner:** the unweighted SVM.

### Experiment 4: Improving Rare-Emotion Classification

Reuse the validation-selected Experiment 3 SVM and change only:

```python
class_weight=None        # Original model
class_weight="balanced"  # Modified model
```

Keep features, C, random seed, label order, partitions, and default decision rules fixed. Reuse the original fitted baseline and verify its test predictions against Experiment 3. Fit the balanced copy using training data only; its vocabulary and IDF must match the original.

For each emotion's binary classifier, balanced weights are:

```text
Positive weight = training_rows / (2 × positive_training_examples)
Negative weight = training_rows / (2 × negative_training_examples)
```

Select the model for the next experiment using validation results. No emotion-specific thresholds are tuned in Experiment 4.

### Leakage Prevention

- Keep records sharing review IDs or normalized sentence text in connected groups.
- Verify group and record separation across partitions.
- Screen long texts for conservative lexical near duplicates using three-word-shingle Jaccard similarity of at least 0.95.
- Fit vocabulary, IDF, and classifiers on training data only.
- Use validation data for configuration selection.
- Use the same test identities and eight target labels for supervised comparisons.
- Retain split manifests, frozen choices, source checks, and predictions for auditing.

The lexical screen is not a guarantee that every semantic paraphrase is detected.

## Results

### Paper Reproduction

| Metric | Published | Recomputed |
| --- | ---: | ---: |
| Mean final human-pair kappa | Approximately 0.69 | 0.694 |
| Combined LLM precision | 70.68% | 70.68% |
| Combined LLM recall | 66.81% | 66.80% |
| Combined LLM paper-compatible F1 | 68.69% | 68.68% |

All **30 per-label percentages in the paper's Table IV** match at the printed precision. Small aggregate differences remain visible in the comparison files.

**Metric convention:** Table II is closely recovered using the harmonic mean of support-weighted precision and recall. The printed human-table F1 columns are recovered using an arithmetic mean of precision and recall, which is not standard F1. These are inferred reproduction conventions; the new supervised experiments use standard F1 throughout.

Historical time and cost are recomputed from the authors' workbook. Its annotation-cost columns exclude separately reported guideline processing. These figures are not current API prices or newly measured bills.

### Experiment 3: Held-Out Baselines

All approaches use the same **163 test records** and eight emotions. Scores are fractions between zero and one.

| Approach | Micro precision | Micro recall | Micro F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Prevalence baseline | 0.2362 | 0.8508 | 0.3697 | 0.1865 |
| Logistic Regression | 0.8226 | 0.2818 | 0.4198 | 0.1907 |
| Linear SVM | 0.7083 | 0.4696 | **0.5648** | **0.2990** |

Both unweighted learned models have zero F1 for the four named rare emotions on this test split, establishing the Experiment 4 baseline.

### Experiment 4: Class-Weighting Comparison

| Metric | Original SVM | Balanced SVM |
| --- | ---: | ---: |
| Micro precision | 0.7083 | 0.6483 |
| Micro recall | 0.4696 | 0.5193 |
| Micro F1 | 0.5648 | 0.5767 |
| Macro F1 | 0.2990 | 0.3408 |
| Rare-emotion macro F1 | 0.0000 | 0.0742 |

| Rare emotion | Positive test examples | Original F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Fear | 2 | 0.0000 | 0.0000 |
| Anger | 4 | 0.0000 | 0.0000 |
| Disgust | 11 | 0.0000 | 0.1429 |
| Surprise | 10 | 0.0000 | 0.1538 |

Weighting improves average F1 and recall, while micro precision falls. It improves Disgust and Surprise but leaves Fear and Anger unresolved. Joy F1 decreases, and Anticipation F1 is unchanged. The full table reports Precision, Recall, and F1 for all eight emotions.

Validation macro F1 also improves from **0.3475 to 0.3862**, so the balanced SVM is selected for Experiment 5 independently of the test results.

## Repository Structure

Use the completion bundle as the repository root. The original replication resources are extracted separately into `emotion_replication/`.

```text
.
├── README.md
├── Emotion_Classification_Project_3_Experiments_3_4_Completed.ipynb
├── Experiment_2_Figures_3_4.py
├── Experiment_3_Ready_To_Paste.ipynb
├── Experiment_3_Complete.py
├── Experiment_3_Tests.py
├── Experiment_4_Ready_To_Paste.ipynb
├── Experiment_4_Complete.py
├── Experiment_4_Tests.py
├── Experiment_4_README.md
├── requirements.txt
├── AUTHORS_LICENSE.md
├── ground-truth.xlsx
├── emotion_replication/              # Extract the original replication ZIP here
│   ├── replication.py
│   ├── supplementary.py
│   └── data/
├── experiment_3_results/
│   ├── split_manifest.csv
│   ├── validation_search.csv
│   ├── frozen_selection.json
│   ├── comparison_overall.csv
│   ├── comparison_per_emotion.csv
│   ├── best_pipeline.joblib
│   └── ...
└── experiment_4_results/
    ├── comparison_all_emotions.csv
    ├── comparison_rare_emotions.csv
    ├── comparison_overall.csv
    ├── validation_comparison.csv
    ├── training_class_weights.csv
    ├── frozen_selection.json
    ├── weighted_pipeline.joblib
    ├── best_after_experiment4.joblib
    └── ...
```

## Getting Started

### Google Colab

1. Open the canonical completed notebook in Colab.
2. Run its cells in order.
3. Supply `ground-truth.xlsx` and the original `emotion_replication_bundle.zip` when the earlier setup cells request them.
4. Review the paper-reproduction sections, then **8.3 Experiment 3** and **8.4 Experiment 4** at the end.
5. Download generated CSVs, plots, and models from the runtime if you want to preserve a new run.

A CPU runtime is sufficient. No GPU, API key, or paid LLM call is required for the completed experiments.

> The Experiment 3/4 completion bundle includes their dataset and saved artifacts. It does not replace the full historical LLM annotation package required by Experiments 1–2.

### Local Python Setup

Use Python **3.11 or 3.12**. Validation was performed with Python 3.12.

From the extracted completion-bundle folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open the notebook with this environment as its Python kernel, or reuse the functions in `Experiment_3_Complete.py` and `Experiment_4_Complete.py`.

### Continue Manually in an Existing Notebook

Copy the cells in this order:

1. `Experiment_3_Ready_To_Paste.ipynb`
2. `Experiment_4_Ready_To_Paste.ipynb`

Experiment 3 requires your existing grouped `train_df`, `val_df`, and `test_df`. It creates:

```python
best_pipeline
exp3_train_df
exp3_val_df
exp3_test_df
EXP3_OUT
```

Experiment 4 uses those variables and creates:

```python
exp4_original_pipeline
exp4_weighted_pipeline
exp4_comparison
exp4_summary
best_pipeline_after_exp4
```

Use the filtered `exp3_*` dataframes for subsequent experiments. Creating the grouped split alone does not retrain earlier models; the completed Experiment 3 section performs that training explicitly.

## Outputs and Saved Models

| Output | Purpose |
| --- | --- |
| `experiment_3_results/split_manifest.csv` | Exact supervised partition identities |
| `experiment_3_results/validation_search.csv` | All 12 configuration results |
| `experiment_3_results/comparison_overall.csv` | Baseline/LR/SVM scores and efficiency measurements |
| `experiment_3_results/comparison_per_emotion.csv` | Per-emotion Precision, Recall, F1, and support |
| `experiment_3_results/best_pipeline.joblib` | Validation-selected unweighted pipeline |
| `experiment_4_results/comparison_all_emotions.csv` | Controlled comparison for all eight emotions |
| `experiment_4_results/comparison_rare_emotions.csv` | Rare-emotion comparison |
| `experiment_4_results/validation_comparison.csv` | Validation scores used for the next model choice |
| `experiment_4_results/training_class_weights.csv` | Training counts and calculated weights |
| `experiment_4_results/best_after_experiment4.joblib` | Validation-selected pipeline for Experiment 5 |
| `experiment_4_results/per_sentence_changes.csv` | Changes in wrong-label counts per sentence |
| `*/frozen_selection.json` | Saved selection rules and choices |
| `*/label_order.json` | Ordered target labels |
| `*/VALIDATION.json`, `*/TEST_RESULTS.txt` | Execution and verification records |

The original `best_pipeline` remains unweighted. The selected balanced model is available as `best_pipeline_after_exp4` and as its saved artifact.

```python
from pathlib import Path
import json
import joblib

folder = Path("experiment_4_results")
pipeline = joblib.load(folder / "best_after_experiment4.joblib")
labels = json.loads((folder / "label_order.json").read_text())

sentences = ["I like the app, but it crashes whenever I open the camera."]
predictions = pipeline.predict(sentences)

for sentence, row in zip(sentences, predictions):
    emotions = [label for label, value in zip(labels, row) if value == 1]
    print(sentence)
    print(emotions or ["No predicted emotion"])
```

Fit time includes the selected model's feature fitting and training, not the complete search budget. Prediction time includes text transformation. Model KiB is serialized file size, not RAM usage. Timings vary across machines.

## Validation

Run the artifact checks from the completion-bundle folder:

```bash
python Experiment_3_Tests.py
python Experiment_4_Tests.py
```

| Component | Executed new code cells | Checks passed |
| --- | ---: | ---: |
| Experiment 3 | 11 | 8 |
| Experiment 4 | 10 | 8 |

Checks cover partition identities, training-only features, validation-based selection, baseline rules, metric calculations, identical test cohorts, class-weight isolation, unchanged baseline predictions, and saved-model compatibility.

New cells were executed locally in IPython. The original Colab upload/mount cells were not rerun locally. No provider API calls were made.

## Limitations

- Rare emotions have very few positive test examples, especially Fear and Anger.
- These are single-split results from a small dataset, not universal model rankings.
- Weighting does not improve every emotion or every metric.
- The 1,274-versus-1,272 label discrepancy remains documented rather than edited away.
- Released LLM outputs are historical annotations, not new blind generations. Their guidelines were developed through the original annotation study.
- Full-dataset ten-label reproduction scores are not directly comparable to eight-label held-out ML scores.
- Exact lexical grouping cannot establish absence of all semantic overlap.
- Local timing and serialized size do not provide a complete memory or deployment-cost analysis.

## Remaining Work

- [ ] **Experiment 5:** optimize a separate threshold for each emotion using validation data.
- [ ] Evaluate LR probability thresholds from 0.10 to 0.90 in 0.05 increments, where applicable.
- [ ] Handle SVM decision-margin thresholds explicitly; raw SVM scores are not probabilities.
- [ ] Apply the proposal's validation precision constraint of at least 40%, recording cases with no feasible threshold.
- [ ] Freeze thresholds before test evaluation and compare default versus tuned P/R/F1.
- [ ] Complete the final same-test, eight-emotion comparison with the released LLM baselines.
- [ ] Consolidate the final research report, error analysis, and uncertainty discussion.

Test outcomes have already been observed during these experiments. Preserve the fixed protocol rather than repeatedly changing configurations or seeds to improve the same test set.

## References and Attribution

### Original Research

Quim Motger, Marc Oriol, Max Tiessler, Xavier Franch, and Jordi Marco. **What About Emotions? Guiding Fine-Grained Emotion Extraction from Mobile App Reviews.** 2025 IEEE 33rd International Requirements Engineering Conference (RE), pp. 6–18.

- [Paper DOI](https://doi.org/10.1109/RE63999.2025.00012)
- [Dataset and replication package](https://doi.org/10.6084/m9.figshare.28548638)

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

### License Notices

The included original authors' materials are accompanied by `AUTHORS_LICENSE.md` (GNU GPL version 3). Preserve upstream attribution and license notices when reusing those resources. The academic project extends and evaluates the published work; it does not claim authorship of the original dataset, annotation guidelines, or LLM annotations.
