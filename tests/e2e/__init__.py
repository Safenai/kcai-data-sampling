"""End-to-end tests: full job runs through the CLI.

These run one real ``SamplingJob`` over the synthetic data from a config dict
through ``cli.run`` — no subprocesses, no separate venvs — and assert on
the on-disk products: the metadata-only ledger and the hashed payload PNGs,
and their exact-bytes invariance across the batch knobs.
"""