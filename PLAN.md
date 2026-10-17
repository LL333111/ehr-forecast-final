# Release plan

The final LightGBM recipe and weights are fixed. This phase simplifies and
verifies the deliverable; it does not tune or retrain the model.

1. Complete: extract the final data, feature, selection, training and prediction pipeline.
2. Complete: verify synthetic boundaries and equivalence to the accepted training features,
   selected columns, predictions and validation metrics.
3. Complete locally: prepare concise English documentation, the technical report,
   aggregate results and model archive; preserve old research in the ignored archive.
4. Pending: upload to the new GitHub repository after its owner supplies the destination.

Remaining submission dependencies: Hugging Face model publication, W&B account
synchronization and the actual run URL, and evaluator-provided hidden-test
metrics. None of these authorize additional model selection or fitting.
