# Output temporário

Use `output/<experiment-id>/<dataset>/` for new runs. Do not use a dataset
directory as an experiment name.

The old roots `output-cifar-10/` and `output-mnist/` were migrated to
`results/`. They must not return as active output roots.

These paths remain because they preserve provenance or historical report data:

- `output/cifar-10/drift-agent-2/` is the imported E07 matrix source.
- `output/cifar-10/severity-calibration/` is the E07 calibration source.
- `output/cifar-10/drift-agent/` is an older E07 output.
- `output/cifar-10/IID_p75.*` and `NON_IID_p75.*` are old E04 plots.
- `output/drift/` and `output/corrupted-drift/` are old E05 plots.
- `output/mnist/` and `output/fashion-mnist/` are old base-simulation outputs.

Do not publish these paths as new experiment results. Use the namespaced
temporary paths and publish validated files in `results/`.
