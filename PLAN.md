# Release plan

The final LightGBM recipe and weights are fixed. This phase simplifies and
verifies the deliverable; it does not tune or retrain the model.

1. Complete: extract the final data, feature, selection, training and prediction pipeline.
2. Complete: verify synthetic boundaries and equivalence to the accepted training features,
   selected columns, predictions and validation metrics.
3. Complete locally: prepare concise English documentation, the technical report,
   aggregate results and model archive; preserve old research in the ignored archive.
4. Complete: publish the 48-file final release to
   https://github.com/LL333111/ehr-forecast-final and verify the remote files.
5. Complete: publish the unchanged final model to
   https://huggingface.co/X13579/ehr-forecast-final and verify every downloaded
   file without authentication; update the documentation and report with the model link.
6. Complete: shorten the English report while retaining the required methods,
   model comparison, curves, condition-level results, AI workflow and limitations.
7. Complete: audit and synchronize the two final W&B runs, verify anonymous
   read access and every curve point, add the report links, and prepare all four
   submission items.
8. Complete: recheck the original task requirements and public artifacts; install
   the packaged code in a fresh environment, rebuild history from raw inputs, and
   reproduce both prediction files and all validation metrics exactly. All 39
   synthetic tests pass. Refresh the report and complete handoff archive.
9. Complete: revise the report's author voice and AI workflow to describe
   individual review, iterative planning, tests and development logs. The three-page
   PDF and all 39 synthetic tests pass review. Public and packaged reports must
   use this verified revision.
10. Complete: fact-check report claims against source code, model contents,
    locked predictions and historical records. Recomputed all 40 validation
    rows and 60 comparison metrics without numerical discrepancies; clarified
    temporal handling and the limits of model-selection estimates. All 39 tests pass.
11. Complete: publish the accepted test predictions as a GitHub release attachment,
    link them from the README, and verify the anonymous download. All 39 tests pass;
    the model and prediction values are unchanged.

The submission files and public links are ready. Hidden-test metrics remain
dependent on the evaluator; they cannot be computed from the released inputs.
No further model selection or fitting is authorized by this release phase.
