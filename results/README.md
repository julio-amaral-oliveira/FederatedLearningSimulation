# Results

The `results/` directory stores published or historical experiment artifacts.

- `e07-drift-agent/` stores the current official E07 publication.
- `e04-static-comparison/` stores historical E04 raw data.
- `e05-temporal-drift/` stores historical E05 raw data and QA data.
- `e06-drift-agent-prototype/` stores legacy E06 data.
- `qa/` stores comparisons that use more than one experiment family.

Do not combine historical data with the official E07 matrix. Read the README in
each result directory before using its files.

The old roots `output-cifar-10/`, `output-mnist/` and `drift-agent/` were
migrated on 2026-07-31. New runs use `output/<experiment-id>/<dataset>/`.
