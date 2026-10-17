# Development log

## 2026-10-16 - Final English release

**Work.** Extracted the final pipeline into one installable package, removed
obsolete model implementations and Torch dependencies, and added a concise
English README, four-page report and aggregate evidence. The native model
weights and fitted preprocessing are unchanged. All 39 synthetic tests pass;
formatting and local editable installation also pass.

**Rationale.** Make the final submission readable and reproducible without
depending on a directory of historical experiments.

**Impact.** The reconstructed training features, labels, prior masks and quality
columns match the accepted final fit exactly. Training prediction replay differs
from the cloud result by at most 1.11e-16. Test and validation CSVs are byte-for-byte
identical, and validation metrics are unchanged. Changing the extraction context
initially caused one near-zero feature to round differently; restoring the accepted
row-local context resolved it without using held-out patients for fitting.

**Limitations.** This verifies the existing pinned Python environment, not a fresh
dependency installation or a new cloud retraining. Official validation was reused
historically. Hidden-test scores are unavailable; Hugging Face publication and a
W&B online link remain pending. Historical research and operational logs are
preserved in the original workspace's ignored archive, not in this release.

**Next.** Upload the reviewed files to the owner's new GitHub repository after its
URL is supplied. Complete the separate model-hosting and run-link requirements
before submission. Do not tune or retrain from these validation results.

## 2026-10-16 - GitHub publication

**Work.** Published the 48-file final release to
[LL333111/ehr-forecast-final](https://github.com/LL333111/ehr-forecast-final).
Updated the repository links and report, checked the remote file tree, and
reran all 39 synthetic tests successfully.

**Rationale.** Provide one public, reviewable version of the accepted pipeline.

**Impact.** Model weights, preprocessing, predictions and reported metrics are
unchanged. Raw data, patient-level outputs and research history remain local;
the public repository starts with the isolated final-release commit.

**Limitations.** Hugging Face publication and the W&B online run link still
need completion. Hidden-test scores remain unavailable.

**Next.** Complete those submission links and use the existing final predictions
and report. This publication does not authorize further model selection or fitting.

## 2026-10-16 - Hugging Face publication

**Work.** Published the final model archive, English model card, attribution notice
and checksums to [X13579/ehr-forecast-final](https://huggingface.co/X13579/ehr-forecast-final).
Downloaded all four files without authentication and verified their SHA-256 hashes.
Added the model link to the README, results metadata and report.
All 39 synthetic tests pass; the four-page report's layout and links were checked.

**Rationale.** Fulfill the model-hosting requirement and let evaluators retrieve
the exact accepted weights and preprocessing.

**Impact.** The model archive is byte-for-byte identical to the existing final
bundle. No patient rows were uploaded, and no model fitting or selection occurred.

**Limitations.** W&B synchronization and the report's online run link remain
pending. Hidden-test outcomes remain unavailable.

**Next.** Complete the W&B link and submit the existing predictions and report.

## 2026-10-16 - Report revision

**Work.** Reduced the report from four pages to three and cut the text by 34%.
Checked all 15 comparison rows, 40 condition results, links and page layouts.
All 39 synthetic tests pass.

**Rationale.** Remove repetition and make the methods and findings easier to read.

**Impact.** Required methods, curves and results remain; model weights,
predictions and metrics are unchanged.

**Limitations.** W&B links are still pending, and hidden-test scores are unavailable.

**Next.** Add the verified W&B links after account synchronization.

## 2026-10-16 - W&B records and submission checks

**Work.** Synchronized the original cloud-training run and the fixed-model
validation replay. Verified anonymous read access and exact agreement with all
150 training points and 16 validation points. Added both links to the three-page
report and README. All 39 synthetic tests pass.

**Rationale.** Complete the required experiment records and submission links.

**Impact.** The report and link list are ready alongside predictions for all
358 test patients and 40 target codes. Model weights and predictions are unchanged;
the W&B records contain metrics and run metadata, not patient rows.

**Limitations.** Validation is a post-training replay on a historically reused
split. Hidden-test metrics remain unavailable until the evaluator provides them.

**Next.** Submit the named report, predictions, public code and model links.
Do not use this release work to tune or retrain the model.
