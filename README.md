# EHR Forecast

Predict first-recorded diagnoses over five years from synthetic patient histories.
The final model uses 512 features and 40 independent LightGBM classifiers.

| Official split | Patients | Use |
| --- | ---: | --- |
| Train | 2,791 | Fit preprocessing, feature selection and models |
| Validation | 365 | Score the fixed model after locking predictions |
| Test | 358 | Generate probabilities; outcomes are hidden |

Final official-validation **mAP: 0.246334; macro AUROC: 0.762251**. These are not
hidden-test scores. Earlier project versions used official validation, so this
is not a previously untouched external evaluation.

## Predict

Use Python 3.12. Place the supplied task files under `data/raw/` without changing
their directory structure. Data is local and excluded from Git.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -e ".[test]"
ehr-forecast predict --raw data/raw --output outputs/predictions.csv
```

The command checks the bundled weights, reconstructs past-only features, and
writes one row per test patient in the task's target-code order. Conditions
already recorded before the anchor receive probability zero. The output path
must be new; cached inputs are checked before reuse.

To reproduce the fixed official-validation score:

```bash
ehr-forecast predict --raw data/raw --split val --output outputs/validation.csv
ehr-forecast evaluate --raw data/raw --predictions outputs/validation.csv --output outputs/validation_metrics.json
python -m pytest
```

Evaluation checks the saved prediction receipt before reading validation
outcomes. It does not fit, tune, calibrate, or select a checkpoint.

## Data and features

One input is a patient's history strictly before their anchor date. For train
and validation, the task defines the anchor as five years before the last
encounter; test anchors are supplied. The future encounter is used only to
derive that required landmark, never as a predictor.

The pipeline combines demographics, encounter and diagnosis history,
medications, procedures, allergies, care plans, immunizations, devices,
numeric and categorical observations, and structured imaging examination facts.
Features describe counts, recency, elapsed time, measurement trajectories,
missingness, visit relationships and clinical interactions.

Annual QALY/DALY/QOLS measures, costs, reason fields, future stop dates, and
unverified DNA, DICOM pixels and ECG payloads are excluded. A past examination
date alone does not certify that an exported payload contains only information
available at that date. Examination facts can still contribute through the
structured tables.

Training produces 36,327 candidate columns, retains 16,181 after quality checks,
and selects 512 using mutual information, shallow-tree gain, sparse logistic
coefficients and subsample stability. All learned steps use official train
only. The selected set contains five explicitly imaging-related features and
no explicitly ECG-related features; this is a feature-name count, not a claim
that all other modality information is irrelevant.

## Model choice and training

The prior fixed comparison covered elastic-net logistic regression, RBF SVM,
random forest, ExtraTrees, LightGBM, XGBoost, CatBoost, EBM, MLP and TabM, plus
three blends. LightGBM had the strongest single-model out-of-fold mAP and the
highest development mAP (0.240847). The preselected blend reached 0.233099 on
development data. These are historical internal-training comparisons, not
official-validation results; their aggregate results are in
[model_comparison.csv](docs/model_comparison.csv).

The final recipe uses 150 trees, 31 leaves, learning rate 0.05, minimum leaf
size 20, L2 penalty 20, 0.8 row/column sampling and base seed 20261016.
There is no ensemble or early stopping. All 2,791 official training patients
were used after the recipe was fixed. Former internal holdouts are therefore
no longer independent evaluations of the final model.

To reproduce training, first read the supplied `data/raw/ReadMe.md`:

```bash
# Local preparation; produces a separate numeric training directory.
ehr-forecast prepare --raw data/raw --output outputs/prepared

# On an authorized RunPod host with this package and the train extra installed:
# Transfer only outputs/prepared/training/, never raw data or validation/test arrays.
ehr-forecast train --prepared /workspace/training --output /workspace/fitted

# After downloading the fitted directory, combine it with local preprocessing:
ehr-forecast export --prepared outputs/prepared --trained outputs/fitted --output model/retrained.zip
```

Install `.[train]` on that host to include W&B. Training saves an actual offline
W&B run; online synchronization requires the owner's account. Re-training may
depend on the recorded library versions and runtime. The supplied weights are
the accepted final fit; cleanup does not retrain them.

## Files and submission

```text
src/ehr_forecast/   Data, feature selection, training, prediction and evaluation
tests/             Synthetic temporal, split, integrity and replay tests
model/final_model.zip  Native weights and fitted preprocessing; no patient rows
docs/              English report and aggregate results
data/              Supplied raw files, local only
outputs/           Generated local artifacts, ignored by Git
```

[The technical report](docs/results.pdf) covers the representation, leakage
policy, model comparison, curves, condition-level results and AI workflow.
Code is published on [GitHub](https://github.com/LL333111/ehr-forecast-final).
The final weights and fitted preprocessing are published on
[Hugging Face](https://huggingface.co/X13579/ehr-forecast-final).
All four published files were downloaded without authentication and matched
the local checksums. The model archive is identical to the GitHub copy.
Submit `predictions.csv` with the report. W&B synchronization and its online
run link remain pending. Hidden-test metrics must come from the evaluator.
Rename the report to the task's requested
`your_name_results.pdf` when submitting.

This release retains the final pipeline. Superseded experiments and full local
research logs are preserved in an ignored archive in the original workspace.
Legacy prefixes inside feature names are stable identifiers required by the
fitted model. The data is synthetic; real-world clinical performance has not
been established. Data attribution is recorded in [NOTICE](NOTICE).
