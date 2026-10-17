# Agent instructions

- Implementation is in `src/`, tests in `tests/`, generated files in `outputs/`.
- `data/` is local, read-only raw input. Never stage it or copy patient rows into tracked files.
- Follow `PLAN.md`; do not start unrelated experiments.
- Never use post-anchor information as a predictor or fit on official validation/test.
- Inspect schemas before using columns. Preserve the fixed feature identifiers and target order.
- Keep the final model immutable. Run tests before declaring a change complete.
- Formal real-data model fitting runs only on an authorized RunPod host. Local synthetic tests are allowed.
- Keep public code, comments and documentation concise and in English.
- Record completed work, rationale, impact, limitations and next steps in `DEVELOPMENT_LOG.md`.
- Keep historical research and private operational records in the local ignored archive.
