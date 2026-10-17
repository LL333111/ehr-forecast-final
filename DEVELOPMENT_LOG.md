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
