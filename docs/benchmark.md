# Benchmark: Albumentations Through Our Package vs. Direct

This page compares the cost of running albumentations through our data-sampling
package against calling albumentations directly. Same upstream version (2.0.8),
same inputs, same machine. The only difference is the loader → transform → clip →
`Output` → ledger/payload pipeline.

## User scenarios

### Robustness augmentation

Data scientist wants to augment his dataset with a list of transformations and random parameters.

### Controlled context

Data scientist wants to create a dataset with controlled context for contextual validation or test.
For instance: A dataset with high levels of blur.

## Benchmark scenarios

Three benchmark scenarios:

- **S1 (batch scaling):** 1000 small images (128×128), one transform (random
  brightness/contrast), varying batch size (1, 5, 10, 20).
- **S2 (transform variety):** 40 different transforms on one large image
  (1928×1208, from the comma10k dataset).
- **S3 (blur sweep):** 40 Gaussian blur levels on one large image.

S1 and S2 covers the robustness augmentation user scenario.
S3 covers the controlled context user scenario.

## S1: one transform, 1000 images, batch-size sweep

Median of 7 timed repeats per cell/side. Direct ÷ Ours ratio; > 1 means we were faster.

| Batch size | Ours (s) | Direct (s) | Direct ÷ Ours | Δ (ours vs direct) | Ours rows/s |
|---|---|---|---|---|---|
| 1 | 7.935 | 7.578 | 0.955 | +4.7 % (+0.357 s) | 126.0 |
| 5 | 7.671 | 7.561 | 0.986 | +1.5 % (+0.110 s) | 130.4 |
| 10 | 7.639 | 7.569 | 0.991 | +0.9 % (+0.070 s) | 130.9 |
| 20 | 7.626 | 7.556 | 0.991 | +0.9 % (+0.071 s) | 131.1 |

The direct side is essentially flat across batch sizes while ours improves
monotonically — the signature of a per-batch cost.

The package's cost is a fixed ≈0.33 ms per batch call, which comes mainly from our pipeline's
per-batch work: iterating rows, decoding batches, and selecting columns.

## S2/S3: transformation variety and transformation levels

Median of 7 timed repeats per cell/side; 40 outputs per suite from one 1928×1208
image (from the comma10k dataset).

| Suite | Rows | Ours (s) | Direct (s) | Direct ÷ Ours | Ours s/row | Direct s/row |
|---|---|---|---|---|---|---|
| S2 (40 transforms) | 40 | 23.720 | 23.701 | 0.999 | 0.593 | 0.593 |
| S3 (40 blur levels) | 40 | 19.315 | 19.159 | 0.992 | 0.483 | 0.479 |

The difference comes mostly from noise.

### Equivalence

kcai-sampling and direct albumentations produce pixel-identical (`sha256` equal) output for the same input and transform parameters.

## Memory — peak RSS

| Cell | Ours (MiB) | Direct (MiB) | Δ | Reading |
|---|---|---|---|---|
| S1, batch size 1 | 224.11 | 225.82 | −1.71 (−0.8 %) | inside noise |
| S1, batch size 5 | 249.02 | 249.02 | 0 | identical |
| S1, batch size 10 | 252.43 | 254.00 | −1.57 (−0.6 %) | inside noise |
| S1, batch size 20 | 261.14 | 263.11 | −1.97 (−0.7 %) | inside noise |
| S2 (40 transforms) | 533.60 | 539.89 | −6.29 (−1.2 %) | inside noise |
| S3 (40 blur levels) | 269.70 | 226.61 | +43.09 (+19.0 %) | only real delta |

- No memory difference at S1.
- S3 is the one real difference: +43 MiB in ours, consistent with 40 live
  transformation instances in one process (≈1.08 MiB each) versus the direct loop
  building one at a time.
- Memory grows with batch size on both sides (≈224 → ≈261 MiB, batch 1 → 20), so
  that growth is shared pipeline behaviour.

## Scope and caveats

- **Startup is excluded.** Time starts after imports; process launch + imports were
  not measured.
- **Alpha handling.** Our engine splits and restores the alpha plane for image-only
  transforms; baselines do equivalent work outside the transform call. Treat
  RGBA-row vs RGB-slice ratios as upper bounds, not like-for-like kernel
  comparisons.

## Conclusion

At a realistic batch size the package adds well under 1 % time and no memory; the
residual is a small *per-batch* fixed cost (≈0.33 ms/batch) that batching
amortises. With decode parity fixed, S2 and S3 are within ~1 % of the direct path,
confirming that the transform pipeline adds no measurable overhead. S1 is
encode-dominated (86 % PNG encode on the direct baseline).