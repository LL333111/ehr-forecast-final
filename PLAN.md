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

Remaining submission dependencies: W&B account synchronization and the actual
run URL, and evaluator-provided hidden-test
metrics. None of these authorize additional model selection or fitting.
