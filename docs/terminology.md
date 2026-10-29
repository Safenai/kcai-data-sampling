# Terminology

Faced with a generated sample, three different questions arise:

| Question | What it names | Term |
| --- | --- | --- |
| What was done? | the operation | transformation |
| By what means? | the family: role of the model in the computation | procedural / generative / adversarial |
| With what effect, on the data? | the regime: is the information the decision needs still there? | augmentation / corruption |
| With what effect, on the annotation? | the state of each field | identical / transported / dropped / invalidated |

The axes are independent. Any means can produce any effect, any effect can be
obtained by any means, and what happens to the annotation follows neither.

The vocabulary does not depend on the modality: *sample* is the generic term
(image, sequence, point cloud, signal), and operations, families and regimes are
defined with no assumption about data content.

## A transformation T is a function that maps one input sample to one output sample

    T : x ↦ x′

- algorithm: the code, which kind of operation is performed
- resolved parameters: validated against the algorithm's schema, complete, defaults filled in
- seed: fixes every random draw the algorithm makes; recorded `None` when it makes none
- model: present or absent; its role defines the family

A transformation is not a procedure waiting for arguments: it is one fully
specified operation.

| Level | What it is |
| --- | --- |
| algorithm | code, independent of any setting |
| declaration | an algorithm, parameter ranges (sweeps), a number of seeds (not implemented yet: one seed per entry) |
| transformation | one resolved point: algorithm, parameters, seed, model |

## Independent, by default

A list of transformations means each one is applied on its own, to the same
source sample: 10 samples × two transformations = 20 outputs. A sequence (run
one transformation's output through the next) is never the default; it is built
by the consumers of this engine, in code or across jobs.

- independent: applied to the same source sample, without feeding another
  transformation's output; that is what a list of transformations means
- output `id`: the derived, deterministic handle of one generated sample, a hash
  of the other fields of the output except the sample itself (parent,
  algorithm, family, arity, reversible, resolved parameters, seed, models);
  lineage joins on it

## What separates the three families is the role of the model in computing the output

| Family | Role of the model | The output is computed… | Examples of algorithms |
| --- | --- | --- | --- |
| procedural | none | from x alone, and from nothing else | rotation, horizontal flip, brightness, Gaussian blur, noise, JPEG, occlusion, resizing |
| generative | tool | by a model that produces the content | diffusion, img2img, inpainting, GAN, style transfer |
| adversarial | target | against a model, whose response enters the computation | FGSM, PGD, C&W, AutoAttack, query-based attacks, adversarial patches |

The family is not entered. It is carried by the algorithm: declared, or by
default derived from the model's role (none → procedural, tool → generative,
target → adversarial). The user never enters it.

## The regimes: not the algorithm's to claim

A regime is established against a task, not computed by the interface. The task
is the question asked of the sample ("is there a vehicle?"), and y\* is the
answer it needs.

| Regime | Definition | Information criterion |
| --- | --- | --- |
| augmentation | the decision-relevant information survives; bits may well be lost, none of them mattered | I(x′ ; y\*) = I(x ; y\*) |
| corruption | part of what the task needs is gone | I(x′ ; y\*) < I(x ; y\*) |

- Processing can only lose: I(x′ ; y\*) ≤ I(x ; y\*) always. The regime says
  whether that inequality is strict.
- The criterion says what is established, not how to compute it: y\* is judged
  rather than computed.
- One case is settled without judgement: a T that is reversible (x is determined
  by x′) is an augmentation, for every task. The interface computes no regime;
  it only freezes `reversible`, declared by the algorithm.
- *Perturbation* names the displacement δ = x′ − x, not a regime.

The regime belongs to the tuple (sample, task, transformation), and is judged
downstream:

| Transformation | Task | Regime |
| --- | --- | --- |
| grayscale | "is there a vehicle?" | augmentation: shape and position are all the question needs |
| grayscale | "is the traffic light red?" | corruption: the one thing the question needs is gone |
| crop 40 %, centred | "is the scene urban?" | augmentation: the environment is still in the frame |
| crop 40 %, centred | count the pedestrians | corruption: the ones near the edges are gone |
| JPEG q = 50 | read the licence plate | depends on the sample, on how large the plate is |

## What is measured, and what it does not settle

Three quantities describe the displacement δ = x′ − x. Someone computes them
afterwards, from x and x′; none of them is on the output.

| Measurement | What it settles | What it does not |
| --- | --- | --- |
| magnitude ‖δ‖ (L∞, L2, PSNR, SSIM) | how far the sample moved | brightness +20 scores PSNR ≈ 22 dB on a sample from which nothing useful was lost; a 15° rotation moves every pixel and loses nothing |
| direction (a region, a frequency band, a resolution) | what kind of thing was destroyed | not whether the decision needed it: that takes the task |
| isotropy (across several draws) | average case or worst case | not who chose the direction: brightness +20 is as anisotropic as an attack |

None of them decides the regime. The regime is a statement about information,
and none of these three measures information.

## Means and effect are independent

Every cell is reachable. The means does not determine the effect:

|  | augmentation: what the decision needs survives | corruption: part of it is gone |
| --- | --- | --- |
| procedural | rotation 15°, flip, uniform noise 1/255 | blur σ = 3, JPEG q = 20, occlusion |
| generative | day → night by diffusion, img2img strength = 0.02 | inpainting that erases an object |
| adversarial | FGSM ε = 2/255, spatial attack (optimised rotation) | PGD ε = 64/255, visible artefacts |

The family does not predict the regime: an adversarial transformation is usually
an augmentation, it leaves intact everything the decision needs and moves only
the model. The regime does not reveal the family: a blur and a destructive
inpainting both produce a corruption, one without a model, one with. Every entry
is read against a declared task; change the task and an entry can cross the line.

## What a transformation does to the annotation

The signature does not change: T : x ↦ x′, and x carries its annotation. The
interface itself records nothing about the annotation (no `y`, no τ); the four
states below are the vocabulary for reading it downstream.

| State | What happened to that field | How it shows |
| --- | --- | --- |
| identical | the transformation does not act on it | y′ = y |
| transported | τ moved it: boxes mirrored, angle negated, keypoints swapped | the new value |
| dropped | the annotated entity is no longer in the sample | the entry leaves the list |
| invalidated | no correct value can be stated any more: the algorithm has no τ for this field type, or the transformation destroyed what made it statable | a sentinel: an assertion, not a hole |

Three of the four are settled by the declaration: identical, transported and
invalidated follow from what the algorithm declares for that annotation type (an
identity, a real τ, or none at all). Only dropped depends on the sample itself.

In the adversarial family, the annotation must stay identical: the prediction
moves, the truth does not. An adversarial transformation that changed
the annotation would have proved nothing.

## τ cannot be derived, it has to be written

τ is a fact about what the annotation means, not about its shape:

| Annotation field | What a horizontal flip requires | τ |
| --- | --- | --- |
| class label | nothing | identity |
| bounding boxes | flip the coordinates | x → W − x |
| segmentation mask | mirror the pixels | mirror |
| steering angle | negate the angle | θ → −θ |
| pose keypoints | mirror and swap the left/right indices | mirror ∘ swap left_\* ↔ right_\* |
| text transcription | nothing correct exists: "b" reads as "d" | none: the field is invalidated |

Bounding boxes and pose keypoints both read as "mirror the coordinates". Do
exactly that to keypoints and every coordinate is right while every annotation
is wrong: `left_wrist` is now on the right. Geometry gives the first half of τ;
only the meaning of the annotation gives the second.

## What a transformation applies to

| Term | What it is | Who fixes it |
| --- | --- | --- |
| sample | the unit element the target function consumes; one id per sample | the data selection |
| data selection | the set of samples a campaign is run on: in the code, the rows a dataloader yields | user configuration |
| batch | an execution split, `(B, *sample)`: carries no meaning, never changes the output | the run, for memory |
| arity | how many samples are read to produce one output | the algorithm |

Only the unary interface, x ↦ x′, is implemented: one output per input sample,
same shape and value range as the input.

## What one output carries, and what it does not

One output = one generated sample. It carries everything that produced it, and
nothing else:

| Group | Fields |
| --- | --- |
| the sample | `x` (x′, in memory) |
| lineage | `parent_id` |
| algorithm (declared, frozen) | `algorithm`, `family`, `arity`, `reversible` |
| replay | `params`, `seed` |
| models | `tool_model`, `target_model` (name, or `NULL`) |
| identity | `id`: a hash of the fields above except `x`, derived |

Every `NULL` is an assertion: no seed, nothing was drawn; no `target_model`, not
adversarial.

This is the output in memory. The ledger row the job writes for it adds where it
was run and stored (`selection`, `dataloader`, `artifact`, the payload file name)
and any source columns passed through with `include`.

The rule is subtraction, not accumulation. A field earns its place only if it
cannot be recovered later:

| | What | Where it is found |
| --- | --- | --- |
| retrievable | x, the annotation | through `parent_id`, in the input selection |
| recomputable | δ, L∞, L2, PSNR, SSIM, the measured budget | a downstream module, from x and x′ |
| judged | the regime, whether the annotation is still statable | downstream, against a task |

`decision_flip`, the evaluated model changing its mind, cannot serve as a
validity indicator. It measures the reaction of the very model we are trying to
probe: using it to decide whether a sample is valid makes the test circular.
Validity is judged without the model under evaluation.
