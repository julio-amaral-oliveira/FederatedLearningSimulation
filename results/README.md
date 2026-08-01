# Results

The `results/` directory stores published or historical experiment artifacts.

- `e07-drift-agent/` stores the current official E07 publication.
- `e04-static-comparison/` stores historical E04 raw data and plots for each
  dataset.
- `e05-temporal-drift/` stores historical E05 raw data, plots and QA data.
- `e06-drift-agent-prototype/` stores legacy E06 data.
- `e07-drift-agent/` also stores the current source provenance and the previous
  E07 matrix under `provenance/` and `legacy/`.
- `qa/` stores comparisons that use more than one experiment family.

Do not combine historical data with the official E07 matrix. Read the README in
each result directory before using its files.

The old roots and the mixed `output/` artifacts were migrated on 2026-07-31.
See [the migration inventory](output-migration-2026-07-31.md) for the SHA-256
record. New runs use `output/<experiment-id>/<dataset>/`.
