# Output temporário

Use `output/<experiment-id>/<dataset>/` for new runs. Do not use a dataset
directory as an experiment name.

The migration on 2026-07-31 removed the historical and provenance artifacts
from this root. The published and preserved copies are in `results/`.

Keep `output/` for temporary execution files. Publish only validated files in
the matching `results/<experiment-id>/<dataset>/` directory.
