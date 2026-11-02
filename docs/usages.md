# Usages

## Analysis scenarios

Use [dqm-ml](https://github.com/Safenai/dqm-ml-workspace) to measure the
diversity, representativeness, or domain gap of a dataset before deciding what
to sample or augment.

### Gap identification

Measure a dataset's diversity/representativeness to identify which contexts or
features are under-covered, then target those gaps with kcai-sampling.

### Domain gap assessment

Quantify the gap between a source dataset and a target domain to decide whether
robustness augmentation or generative completion is needed.

## Procedural scenarios

We think that in order to build a trustable AI component, model preprocessing should be decorrelated from sampling.
That's why our packages don't provide preprocessing functions, only transformations that have meaning for robustness and contextual evaluations.

### Robustness augmentation

Augment a dataset with a list of transformations and random parameters.

### Controlled context

Create a dataset with controlled context for contextual validation or testing.
For instance: a dataset with high levels of blur.

## Generative scenarios

### Data completion and synthesis

Generate plausible content in missing regions (e.g., inpainting) or synthesize variations to expand coverage of under-represented cases.

### Domain transfer / style adaptation

Generate samples in a target domain or style while preserving semantic content, for robustness testing.

## Adversarial scenarios

### Adversarial robustness evaluation

Generate adversarial examples (e.g., with FGSM) to evaluate model robustness against worst-case perturbations.

### Targeted attack analysis

Assess sensitivity to specific classes or failure modes using targeted adversarial examples.

## Composite scenarios

### Defense evaluation (adversarial + remediation)

Test a defense mechanism against adversarial attacks: generate adversarial samples (adversarial family), apply a small perturbation (generative or procedural transformation with small delta) to "fix" them, then compare model performance across original, adversarial, and fixed samples. The expectation is that adversarial samples decrease performance while fixed samples get as close as possible to the original samples in terms of model performance.

### Multi-family robustness stress test

Combine procedural transforms that act as corruptions with generative edits or adversarial examples to evaluate model behavior under realistic compound shifts.

### Targeted data completion (analysis → sampling)

A data scientist runs DQM-ML metrics on a dataset, identifies under-covered
contexts, then uses kcai-sampling (generative or procedural) to complete those
gaps and re-measures to confirm improvement.

### Domain adaptation pipeline (analysis → sampling)

A data scientist measures the domain gap between source and target, selects the
appropriate family (procedural for robustness, generative for style transfer),
applies it, and re-runs DQM-ML to validate that the gap narrowed.
