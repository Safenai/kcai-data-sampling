# Data sampling

What - the operation:

- transformation

Why - the families:

- procedural
- generative
- adversarial

How - the regimes:

- perturbation
- augmentation
- corruption

# Terminology

## A transformation T is a function that maps one input sample to one output sample

- algorithm: the code, which kind of operation is performed
- resolved parameters: the settings that fix the operation's behaviour
- seed: fixes every random draw the algorithm makes
- model: present or absent; its role defines the family

## What separates the three families is the role of the model in computing the output:

| Family | Role of the model | The output is computed… | Examples of algorithms |
| --- | --- | --- | --- |
| procedural | none | from x alone, and from nothing else | rotation, horizontal flip, brightness, Gaussian blur, noise, JPEG, occlusion, resizing |
| generative | tool | by a model that produces the content | diffusion, img2img, inpainting, GAN, style transfer |
| adversarial | target | against a model, whose response enters the computation | FGSM, PGD, C&W, AutoAttack, query-based attacks, adversarial patches |

## A regime is not declared. It is observed, after the fact, on the input / output pair:

| Regime | Definition | Examples |
| --- | --- | --- |
| perturbation | bounded, sub-perceptual change, defined by a budget ε | FGSM ε = 2/255, uniform noise 1/255 |
| augmentation | visible change, information preserved | rotation 15°, flip, brightness +20, day → night |
| corruption | visible change, information destroyed | blur σ = 3, JPEG q = 20, occlusion, fog, brightness +100 |

## What a transformation does to the label:

Changing the label is not a failure. Transporting it along with the sample can be the point of the transformation.
The failure is a label that should have moved and did not.

| Status | Definition | What T must return |
| --- | --- | --- |
| invariant | y′ = y, the label carries over untouched | x′ |
| covariant | y′ = τ(y), the label is transported by a map τ the algorithm knows | x′ and y′ |
| undefined | no correct y′ exists, the sample has become unlabelable | nothing, discarded |
| unsupported | no τ exists for this label type, the pair is invalid | nothing, campaign refused before running |

- This axis is orthogonal to the regime. Rotation-on-boxes and inpainting-erasure are both covariant, and they fall on either side of the augmentation / corruption line: one is invertible, the other destroys. Neither is invalid.
- In the adversarial family, covariance is a contradiction. An attack is only meaningful if y′ = y: the prediction moves, the truth does not. An adversarial transformation that changed the label would have proved nothing.
- Undefined is not a declared value. The algorithm declares invariant or covariant; whether a given output is still labelable is read.

## Invariant or covariant depends on the label

| Label | Effect of a horizontal flip | Status | τ |
| --- | --- | --- | --- |
| class label | untouched | invariant | identity |
| bounding boxes | flipped coordinates | covariant | x → W − x |
| segmentation mask | mirrored pixels | covariant | mirror |
| steering angle | negated angle | covariant | θ → −θ |
| pose keypoints | mirrored & swapped left/right | covariant | mirror and swap left_* ↔ right_* |
| text transcription | invalidated syntax (e.g. "b" → "d") | unsupported | none |

## Equal magnitude, different meaning

| The pair | What separates them | Criterion | It qualifies |
| --- | --- | --- | --- |
| an occlusion / a global veil (losing a region ≠ degrading everywhere) | support | fraction of coordinates touched (global / local / sparse) | a result (read off one pair) |
| a blur / a rotation (impoverished data ≠ displaced data) | information | I(x ; x′) compared to H(x) | the transformation |
| random noise / an attack (average case ≠ worst case) | isotropy | spectrum of Cov(δ) across seeds (flat or concentrated) | the law (across all its draws) |

## What a transformation destroys

| Mechanism | What happens | Examples |
| --- | --- | --- |
| bijection | T is invertible, δ may be large, nothing is lost | rotation, flip, permutation, brightness below clipping |
| injection | T adds a δ that cannot be separated from x inside x′ | Gaussian noise |
| projection | T collapses dimensions, or compresses below quantisation | blur, occlusion, downsampling, JPEG |

## Means and effect are independent

Every cell is reachable. The means does not determine the effect:

|  | perturbation | augmentation | corruption |
| --- | --- | --- | --- |
| procedural | uniform noise ε = 1/255 | rotation 15°, flip | blur σ = 3, JPEG q = 20 |
| generative | img2img strength = 0.02 | day → night via diffusion | inpainting masking object |
| adversarial | FGSM ε = 2/255 | spatial attack (rotation) | PGD ε = 64/255 |

The family does not predict the regime: adversarial is not necessarily perturbation.
The regime does not reveal the family: blur and inpainting both produce corruption.
The diagonal is only common habit, not a structural rule.

## Declared, measured, or judged, and how much each is worth

|  | Nature | Where it lives | How it is established |
| --- | --- | --- | --- |
| family | property of the algorithm | carried by the algorithm | declared, consistency checkable without running |
| label effect | resolved from the pair (algorithm, label type) | both declare, the campaign resolves | declared, the interface checks that a τ exists, never that it is right |
| regime | half measured, half declared | column of the output table | budget: measured. information: inherited from the declared mechanism |
| validity, is y still statable? | a judgement on the result | column of the output table | an oracle, a human or a rule. Never derived, and never from the evaluated model |

decision_flip : the evaluated model changing its mind, cannot serve as a validity indicator.
It measures the reaction of the very model we are trying to probe: using it to decide whether a sample is valid makes the test circular.
Validity is judged without the model under evaluation.