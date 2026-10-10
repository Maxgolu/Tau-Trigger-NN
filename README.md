# TAUminator: Compact Neural Networks for Tau Triggering

This repository is the complete university submission by Max Golubkov and
Tamar Averbuch. It contains the full development of the project: the original
single-object tau-classification framework, the subsequent event-level pair
study, the unified report and presentation, and the reproducibility material
used to verify the submitted results.

## Submission contents

- **Single-object study:** compact neural classifiers, input-representation
  studies, loss functions, operating-point calibration and evaluation tools.
- **Pair-model study:** joint processing of two same-event trigger objects,
  event-balanced training, event-level calibration and comparison with the
  deterministic measured-pT baseline.
- **Report:** editable LaTeX source and the submitted PDF under [`paper/`](paper/).
- **Presentation:** the delivered presentation and supporting research-history
  material under [`presentation/`](presentation/).
- **Reproducibility:** a data-free CPU demonstration, automated submission
  verifier, tests, frozen configurations and compact result artifacts.

The repository preserves the controlled experiments that led to the submitted
systems. They provide research provenance; the frozen submission artifacts and
results identified below are authoritative.

## Video demonstration

[![Watch the project demo on YouTube](https://i.ytimg.com/vi/ZXKG2rvxmxY/hqdefault.jpg)](https://youtu.be/ZXKG2rvxmxY)

[Watch the project demo on YouTube](https://youtu.be/ZXKG2rvxmxY) (1 min 48 s).
[Download the MP4](demo/demo-atlas-tau.mp4) (1080p, 7.2 MB).

## Project overview

The project studies compact neural networks for tau identification in the
Level-1 trigger. The first stage develops and evaluates classifiers for
individual trigger objects (TOBs). The second stage asks whether processing two
TOBs from the same event can improve signal-event acceptance while keeping the
background-event false-positive rate at or below 0.5%.

The submitted pair-model event pipeline is fixed:

1. Rank finite TOBs within each event by measured `tob_pt`.
2. Retain every finite TOB; exact pT ties use increasing `tob_index`.
3. Form every unordered same-event pair, with no synthetic pair.
4. Give each pair one model score.
5. Use the maximum pair score as the event score.
6. Combine the neural decision with the calibrated deterministic pT branch.

Measured `tob_pt` remains in its raw source unit, MeV, until plotting or
evaluation code explicitly converts it.

## Single-object framework

The first stage of the project studies compact classifiers for individual
TOBs. Its implementation remains the foundation of the repository:

- [`src/model.py`](src/model.py), [`src/features.py`](src/features.py) and
  [`src/losses.py`](src/losses.py) define the models, representations and loss
  functions;
- [`src/train.py`](src/train.py), [`src/evaluate.py`](src/evaluate.py) and
  [`src/checkpoint_selection.py`](src/checkpoint_selection.py) implement
  training, evaluation and model selection;
- [`src/operating_point.py`](src/operating_point.py) provides the shared
  operating-point calibration used by both stages;
- the original configuration families under [`configs/`](configs/) preserve
  the controlled single-object experiments described in the report.

The unified paper gives the motivation, experimental progression and results
for this stage before introducing the event-level pair model.

## Pair-model extension and submitted configuration

The submitted candidate is `all_finite_equal_event`, configured in
[`configs/pair_final_all_finite_equal_event/config.json`](configs/pair_final_all_finite_equal_event/config.json).
It uses the shared two-channel high-resolution EM2 network with 4,357 trainable
parameters. Training excludes mixed-label pairs, applies the p=-1 positive-pair
weight, and gives each eligible event equal weight by averaging its complete
set of pair losses before averaging across events. Inference uses all finite
pairs without truth filtering.

On the final test population, the three-seed mean inclusive efficiency was
**46.70%**, compared with **34.30%** for the deterministic second-highest-pT
baseline: an improvement of **12.40 percentage points**. The complete compact
result is in
[`results/final_test/final_result_summary.json`](results/final_test/final_result_summary.json).
All three measured background false-positive rates were slightly above the
predeclared 0.5% ceiling (0.5183%--0.5223%), so the stricter conjunctive final
success rule did not pass even though efficiency improved substantially.

Run the data-free demonstration with:

```bash
./run_demo.sh
```

The complete editable joint paper and submitted PDF are under [`paper/`](paper/).
They present the single-object study and pair-model extension as one research
project, followed by shared conclusions, future directions and reproducibility
material.

The detailed 47-slide pair-study presentation, including its original speaker
notes and public-reference lists, is preserved at
[`presentation/pair_model_research_history.pptx`](presentation/pair_model_research_history.pptx).
It records the controlled research path and earlier validation stages. Use
[`docs/PRESENTATION_EVIDENCE_AUDIT.md`](docs/PRESENTATION_EVIDENCE_AUDIT.md)
to distinguish that historical evidence from the frozen final-test result, and
[`docs/CODE_TO_PRESENTATION.md`](docs/CODE_TO_PRESENTATION.md) for the
claim-level code, report, and slide crosswalk.

## Pair-model development history

The table below preserves the strongest validation-stage configurations that
led to the final system. They are supporting experiments, not alternatives to
the frozen submitted candidate.

Here, high-resolution means that the network processes each TOB's detailed
12x12 EM2 calorimeter image. The current leader supplies both the absolute
image and a signed, energy-normalized copy to one shared member CNN.

| Role | Input and architecture | Event decision | Mean validation efficiency |
| --- | --- | --- | ---: |
| Current overall leader | Shared two-channel EM2 CNN with coarse cells, shower fraction and measured pT | Neural score OR direct pT branch | 44.57% |
| Deeper compatible encoder | Same inputs with two convolutional layers | Neural score OR direct pT branch | 44.34% |
| Earlier neural-only comparison | One-channel high-resolution network | Maximum neural pair score | 44.09% |
| Compact comparison | Shared MLP with EM2 dominance, shower fraction and measured pT | Maximum neural pair score | 42.46% |

For each seed, efficiency is the number of validation signal-sample events
that pass the event decision divided by all 7,307 validation signal-sample
events. The reported mean is the arithmetic average across seeds 42, 123 and
456. Each seed receives its own threshold, calibrated so that the validation
background-event false-positive rate is at most 0.5%.

An event passes when its final event score satisfies its calibrated decision
rule. The values in this table are adaptive validation evidence; the final
test evidence is reported separately above.

The exact configurations are under
[`configs/pair_highres_two_channel_pt_or/`](configs/pair_highres_two_channel_pt_or/),
[`configs/pair_deeper_shared_encoder/`](configs/pair_deeper_shared_encoder/),
[`configs/pair_highres_powerlaw_pminus1/`](configs/pair_highres_powerlaw_pminus1/)
and
[`configs/pair_rep_raw_shower_pt_context/`](configs/pair_rep_raw_shower_pt_context/).
The compact numerical summary is
[`results/current_configurations.csv`](results/current_configurations.csv).

## How to read an experiment

Each configuration combines five separate choices. Changing one choice creates
a new experiment, but does not necessarily create a new neural architecture.

### Input representation

This defines the information supplied to the network, such as coarse cells,
the high-resolution EM2 image, shower summaries or measured pT. The
implementations are in [`src/pair_features.py`](src/pair_features.py) and
[`src/prepare_pair_data.py`](src/prepare_pair_data.py). The corresponding
experiment groups are indexed in [`configs/README.md`](configs/README.md).

### Neural architecture

This defines how the network processes the inputs, for example with a shared
MLP or a shared CNN. The implementations and model factory are in
[`src/pair_models.py`](src/pair_models.py).

### Training method

This includes the target, loss, pair weighting, optimizer and checkpoint
schedule. Training is implemented in [`src/pair_train.py`](src/pair_train.py),
while each run records its exact choices in a JSON file under
[`configs/`](configs/).

### Event decision

This defines how pair scores become one event decision. The standard neural
decision uses the maximum pair score. Some controlled studies add a direct pT
branch. Pair-to-event reconstruction is in
[`src/pair_data.py`](src/pair_data.py), and the optional combined decision is
in [`src/pair_classifiers.py`](src/pair_classifiers.py).

### Calibration

This defines how thresholds are chosen at a background-event false-positive
rate no greater than 0.5%. The shared threshold rule is in
[`src/operating_point.py`](src/operating_point.py), and cross-fitted checkpoint
evaluation is in [`src/pair_evaluate.py`](src/pair_evaluate.py).

## Core pair components

### Pair workflow

1. Build the two member inputs for one same-event pair.
2. Process both members with the same member encoder when that representation
   uses shared processing.
3. Combine the member outputs and any declared event context in a pair head,
   producing one pair score.
4. Take the maximum score over all pairs in the event.
5. Compare that event score with the model's independently calibrated event
   threshold.

Changing the representation changes the information supplied for each TOB. The
shared encoder converts each TOB into a small learned summary. The
pair-combining network uses the two summaries to produce one pair score.
`src/pair_data.py` handles pair construction and event scoring, while
`src/pair_validate.py` runs the validation procedure.

### Deterministic baseline

The baseline event score is the second-highest selected TOB pT. Pair
construction provides an exact equivalent:

```text
pair score  = min(pT_i, pT_j)
event score = max(pair scores)
```

For every event with at least two finite TOBs:

```text
max(min(pT_i, pT_j)) = second-highest pT
```

Events with fewer than two finite TOBs remain in event-level denominators and
receive a non-passing score of negative infinity.

### Operational labels and populations

Each TOB retains the original tau/background label. A pair is positive when
both of its TOBs are tau-labelled. The code also supports the corresponding
three-class label:

```text
three-class label = label_i + label_j       # 0, 1, or 2
binary label      = 1 when both labels are 1
```

The stored labels do not provide separate generator-particle identities.

The audited source inventory contains:

| Population | Events |
| --- | ---: |
| Signal sample | 72,404 |
| Background sample | 249,022 |
| Total | 321,426 |

The earlier top-four inventory contains 1,292,919 training-split pairs. The
final all-finite training cache contains 2,709,920 eligible same-label pairs
from 222,709 events, including 27,054 positive pairs. The final training view
retains every finite TOB, forms every unordered pair, and removes mixed-label
pairs only from training. Validation and test inference retain all finite
pairs without truth filtering.

### Input representations

The pair modules support the representations implemented in this branch:

| Representation | Pair input |
| --- | --- |
| pT-only control | Higher and lower measured member pT |
| Raw coarse cells | 45 cells from each member, 90 values |
| Raw cells with pT | Member-local cells and pT, 92 values |
| Layer summaries | 16 summaries from each member, 32 values |
| Layer summaries with pT | Member-local summaries and pT, 34 values |
| High-resolution EM2 | Shared processing of each member's absolute and signed-normalized 12x12 EM2 images and 47 scalar values |

The 47 high-resolution member scalars are the 45 coarse cells, the member's
measured pT, and its strongest-3x3 EM2 energy fraction. The event's
second-highest measured pT is appended once at the pair head, not once per
member.

Controlled studies of input representation, event context, output formulation,
fusion and final event decisions are indexed in
[`configs/README.md`](configs/README.md). Their compact validation evidence is
described in [`results/README.md`](results/README.md).

### Models and classifier decisions

`src/pair_models.py` contains the principal models implemented for the pair
study:

- `PairPtMLP`: `2 -> 8 -> 1`, 33 trainable parameters.
- `FlatRawPairMLP`: `90 -> 32 -> 16 -> 1`, 3,457 parameters.
- `SharedHighResolutionPairModel`: configurable one- or two-layer shared CNN
  member encoders followed by a pair head. The current two-image-channel,
  47-scalar/one-context form uses `33 -> 32 -> 16 -> 1` and has 4,357
  parameters. The older one-image-channel form has 4,321 parameters. The pair-sum/balance form
  uses 46 member scalars and three context values, giving
  `35 -> 32 -> 16 -> 1` and 4,369 parameters.

The signed normalized EM2 channel is
`E / max(sum(abs(E)), 1 MeV)`. It describes shower shape while the absolute
channel retains the energy scale.

### Latest controlled conclusions

Five source-removal experiments and four clean representation comparisons were
run with the same validation procedure. Every source-removal model remained
above the deterministic baseline, but none beat the complete two-channel
leader. The closest ablation removed only the engineered best-3x3 fraction and
was 0.20 percentage points lower. The four clean representations were also
lower, so none was promoted.

A compatible two-convolution shared encoder was then tested with the leader's
inputs and pair interface held fixed. It reached 44.34% mean inclusive
validation efficiency, 0.23 percentage points below the one-convolution
leader, and was not selected. Compact summaries are in
[`results/source_ablation/`](results/source_ablation/),
[`results/clean_representation/`](results/clean_representation/) and
[`results/shared_encoder/`](results/shared_encoder/).

`src/pair_classifiers.py` provides the joint-calibration primitive for combining
a neural pair decision with a measured-pT decision under one background budget:

```text
neural pair score >= t  OR  lower pair-member pT >= C
```

For one predeclared pT-branch budget, the pT threshold `C` is calibrated first
and the neural threshold `t` receives the remaining event-level allowance.
Events accepted by both branches count once. The standard `pair_validate.py`
workflow is deliberately NN-only; it cannot silently label that result as an
OR study.

### Validation and calibration

Validation events are divided into deterministic folds A and B within five
event strata. For each model seed and checkpoint:

1. Calibrate an event threshold on background fold A.
2. Evaluate that threshold on fold B.
3. Swap the folds.
4. Pool the held-out integer counts.
5. Select one checkpoint per seed using pair-eligible signal efficiency.
6. Recalibrate the selected seed model on the complete validation background.

Every threshold uses the same rules:

- The denominator is background events, not pairs.
- The empirical event FPR must be no more than 0.5%.
- An event passes when its finite score is greater than or equal to the threshold.
- Tied scores stay together.
- Zero accepted events are allowed.
- Baseline and learned-model thresholds are independent.

The current tracked results are validation results. The reserved test sample is
not used for model or checkpoint selection. If historical pair work used the
same test events, a genuinely untouched external or future sample is required
for the final unbiased evaluation.

## Repository structure

```text
src/
  model.py              single-object neural architectures
  features.py           single-object input representations
  losses.py             weighted and constrained training objectives
  train.py              single-object training
  evaluate.py           single-object evaluation
  operating_point.py    shared threshold calibration
  pair_data.py          deterministic selection, pairing, labels and aggregation
  pair_features.py      pair representations
  pair_context_features.py observable event/pair pT summaries
  pair_models.py        pair neural networks
  pair_train.py         training-only model runner
  pair_predict.py       pair inference and event reconstruction
  pair_validate.py      all-checkpoint validation for one config family
  pair_evaluate.py      cross-fitted checkpoint evaluation
  pair_energy_profiles.py conditional matched-member regional summaries
  pair_classifiers.py   measured-pT plus neural OR calibration
  pair_plots.py         result plots
  prepare_pair_data.py  pair-array preparation from the source CSV/NPZ files

configs/
  cnn_*/
  mlp_*/
  constrained_*/
  pair_pt_control/
  pair_raw_cells/
  pair_high_resolution_em2/
  pair_rep_*/
  pair_gain_*/
  pair_output_*/
  pair_fusion_*/
  pair_highres_*/
  pair_context_*/
  pair_raw_cell_or/

results/
  pt_control/
  training_feature_analysis/
  representation_study/
  high_resolution/
  high_energy_recovery/
  pt_factorial/
  pt_context/
  high_resolution_pt_or/
  pair_or/
  plots/

tests/
  test_pair_*.py
```

The repository contains both the single-object modules and the later pair-model
modules. Both use the same separation of data preparation, features, models,
training, calibration, evaluation and plotting.

## Data and dependencies

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python submission/verify_submission.py
```

The pinned environment targets Python 3.11, matching the verified development
and CI runtime.

The last command verifies the source/config/result hashes, recomputes the
three-seed summary, and runs the data-free demo. It needs no private data,
checkpoints, network access, or GPU. `make verify` additionally runs the
complete test suite after setup.

Place the externally supplied files under a data directory with this layout:

```text
Signal/signal_combined.csv
Signal/signal_combined.npz
Background/bkg_combined.csv
Background/bkg_combined.npz
```

Raw data, checkpoints, predictions and generated caches are intentionally not
tracked by Git. The `prongs` field is not used as a model input.

The data-free demo above is completely self-contained. Exact full retraining of
the frozen final candidate additionally requires the original detector dataset
and the five small parent-normalization files named in the final configuration.
Place those externally supplied, hash-verified files under:

```text
external_artifacts/final_model/train_normalizer.json
external_artifacts/final_model/parent_absolute_mean.npy
external_artifacts/final_model/parent_absolute_scale.npy
external_artifacts/final_model/b01_normalized_mean.npy
external_artifacts/final_model/b01_normalized_scale.npy
```

The trainer refuses missing or hash-mismatched moments. It does not silently
refit normalization on the all-finite cache, because that would no longer
reproduce the reported final model.

## Example workflow

Prepare the pT-only training pairs:

```bash
python src/prepare_pair_data.py \
  --data_dir /path/to/data \
  --split train \
  --representation pair_pt \
  --output pair_cache/pair_pt_train.npz
```

For a configured compact shared-member representation, prepare the arrays from
the configuration itself so the declared member-feature order is preserved:

```bash
python src/prepare_pair_data.py \
  --data_dir /path/to/data \
  --split train \
  --config configs/pair_rep_raw_shower_pt_context/pair_rep_raw_shower_pt_context_s42.json \
  --output pair_cache/raw_shower_pt_context_train.npz
```

The same config-driven command prepares a context-study candidate. For
example, this form changes only the event context to total measured event pT:

```bash
python src/prepare_pair_data.py \
  --data_dir /path/to/data \
  --split train \
  --config configs/pair_context_event_total_pt/pair_context_event_total_pt_s42.json \
  --output pair_cache/context_event_total_pt_train.npz
```

The supported compact member features are measured member `tob_pt`, the event's
second-highest measured `tob_pt`, core-EM2 raw or safely normalized dominance,
and the verified strongest-3x3 EM2 fraction. Both pair members use the same
declared feature order and the same training-fitted normalizer.

The two energy-weighted high-resolution configurations additionally request
`member_min_truth_pt_gev`. The preparation command derives this training-only
analysis coordinate from the two TOB-associated `truth_pt` values and fails
clearly if either value is unavailable for a positive pair. It is used only as
a loss weight, is never a model input, and does not claim that the two TOBs are
matched to two distinct generator-level tau particles. Config-driven validation
and test preparation omit this field automatically; the lower-level API rejects
an explicit request to attach it to either holdout split.

Train one configured seed, then repeat with the seed-123 and seed-456 configs:

```bash
python src/pair_train.py \
  --config configs/pair_pt_control/pair_pt_control_s42.json \
  --pair_data pair_cache/pair_pt_train.npz
```

Prepare validation arrays using the same representation. Then score each saved
checkpoint, recording its seed, checkpoint identity and validation BCE:

```bash
python src/pair_predict.py \
  --config configs/pair_pt_control/pair_pt_control_s42.json \
  --pair_data pair_cache/pair_pt_validation.npz \
  --normalizer experiments/pair_models/pair_pt_control/seed_42/normalizer.npz \
  --checkpoint experiments/pair_models/pair_pt_control/seed_42/checkpoint_epoch_20.pt \
  --model_seed 42 \
  --checkpoint_id checkpoint_epoch_20 \
  --output_dir experiments/pair_models/predictions/seed_42_epoch_20
```

`pair_predict.py` is useful for inspecting one checkpoint. The normal validation
workflow scores every configured checkpoint and seed automatically:

```bash
python src/pair_validate.py \
  --config_dir configs/pair_pt_control \
  --pair_data pair_cache/pair_pt_validation.npz \
  --output_dir experiments/pair_models/pair_pt_control_validation
```

The validation command writes per-checkpoint pair/event predictions, the
cross-fitted checkpoint table, and a selection report containing one selected
checkpoint and one full-validation threshold per seed, plus the independently
calibrated baseline. Validation BCE is calculated from the saved logits; it is
never typed into the selection workflow by hand.

Regenerate the compact result figures:

```bash
python src/pair_plots.py --plot all
```

## Results and scope

`results/` contains the compact final-test summary plus tables and regenerable
figures for the earlier controlled studies. See `results/README.md` for the
available evidence and its interpretation limits.

Completed evidence includes the deterministic mechanics, measured-pT-only
diagnostic, representation and high-resolution studies, validation-only
recovery, OR, pT-factorial and pT-context comparisons, and the frozen final
test evaluation. Earlier adaptive studies remain labeled as validation
evidence and are not presented as independent final results.

### Tracked artifacts

The branch retains more than only the best current configuration. It keeps
reusable implementations and readable per-seed configurations for controlled
experiments that support the reported conclusions, including scientifically
useful negative results. This makes it possible to reproduce why a direction
was retained or rejected.

The repository intentionally excludes raw data, prepared caches, checkpoints,
pair/event prediction tables, temporary execution manifests and private
research notes. Compact result tables and regenerable figures are kept in
`results/` instead.

No binding hardware limit is applied to the research comparisons. The code
reports model input widths and parameter counts so deployment constraints can
be assessed when they are available.

## Tests

Run the focused pair tests:

```bash
python -m unittest discover -s tests -p 'test_pair_*.py'
python -m unittest tests.test_prepare_pair_data
```

Run the complete inherited and pair-model test suite:

```bash
python -m unittest discover -s tests
```

The tests cover same-event pairing, top-four and all-finite populations,
deterministic ties, label construction, equal-event reduction and complete
event batching, raw and engineered feature mappings, model dimensions,
maximum event aggregation, no-pair behavior, independent calibration,
cross-fitted validation and training-only normalization.

Some inherited single-object integration tests load pretrained experiment
weights. Those checks are skipped when the ignored external `experiments/`
artifacts are not present; their configuration and architecture unit tests
still run.
