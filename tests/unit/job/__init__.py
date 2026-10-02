"""Unit tests for the job package: writers, loader wiring, ledger columns.

All five groups built so far use the shared fixtures' synthetic data, the
inline config builders, and — for the job-level tests — the same objects
the CLI assembles: real registry-resolved loaders with real writers, so a
run is one ``SamplingJob`` start to finish.
"""
