# Pair-Based Tau Trigger Classifier

## Overview

This branch extends the single-object tau-classification framework with an
event-level pair model. The pair study asks whether two TOBs can be processed
together to improve signal acceptance while keeping the background-event false
positive rate at or below 0.5%.

The event pipeline is fixed:

1. Rank finite TOBs within each event by measured `tob_pt`.
2. Keep at most the four highest-pT TOBs.
3. Form every unordered same-event pair, up to six pairs per event.
4. Give each pair one model score.
5. Use the maximum pair score as the event score.
6. Calibrate every learned model independently from the deterministic baseline.

Measured `tob_pt` remains in its raw source unit, MeV, until plotting or
evaluation code explicitly converts it.

## Architecture in one pass

1. Build the two member inputs for one same-event pair.
2. Process both members with the same member encoder when that representation
   uses shared processing.
3. Combine the member outputs and any declared event context in a pair head,
   producing one pair score.
4. Take the maximum score over all pairs in the event.
5. Compare that event score with the model's independently calibrated event
   threshold.

The representation changes step 1; the member encoder and pair head define the
learned model. `src/pair_data.py` and `src/pair_validate.py` implement the
pair-to-event and validation contracts.

## Deterministic baseline

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

## Operational labels and populations

Each TOB has a verified binary operational label. A pair receives:

```text
three-class label = label_i + label_j       # 0, 1, or 2
binary label      = 1 when both labels are 1
```

These labels count positively labelled TOB rows. The available export does not
contain a unique generator-particle identity, so a label-2 pair must not be
described as two verified distinct generator-level taus.

The audited source inventory contains:

| Population | Events |
| --- | ---: |
| Signal sample | 72,404 |
| Background sample | 249,022 |
| Total | 321,426 |

The complete top-four inventory contains 1,292,919 training-split pairs. The
eligible training view contains 1,143,864 pairs: every available background
pair and signal pairs from events whose selected top four contain at least two
positive TOB labels.

## Representations in this branch

The pair modules support the representations implemented in this branch:

| Representation | Pair input |
| --- | --- |
| pT-only control | Higher and lower measured member pT |
| Raw coarse cells | 45 cells from each member, 90 values |
| Raw cells with pT | Member-local cells and pT, 92 values |
| Layer summaries | 16 summaries from each member, 32 values |
| Layer summaries with pT | Member-local summaries and pT, 34 values |
| High-resolution EM2 | Shared processing of each member's 12x12 EM2 image and 47 scalar values |

The 47 high-resolution member scalars are the 45 coarse cells, the member's
measured pT, and its strongest-3x3 EM2 energy fraction. The event's
second-highest measured pT is appended once at the pair head, not once per
member.

The current controlled context study keeps those calorimeter inputs fixed and
changes only the measured-pT description supplied to the pair model. Its seven
predeclared alternatives are: total event pT, maximum or total pT outside the
pair, the pair's fraction of event pT, top-four pT concentration, top-four pT
entropy, and pair sum plus balance. Entropy here is the Shannon entropy, in
nats, of the top-four measured-pT fractions; it is the precise implementation
of the deck's informal “top-four pT spread” label. The final alternative removes the two
member-pT scalar slots and supplies pair sum and balance at the pair head. The
implementation is in `src/pair_context_features.py`; each alternative has
separate seed-42, seed-123 and seed-456 configurations under
`configs/pair_context_*/`.

## Models and classifier decisions

`src/pair_models.py` contains the principal models implemented for the pair
study:

- `PairPtMLP`: `2 -> 8 -> 1`, 33 trainable parameters.
- `FlatRawPairMLP`: `90 -> 32 -> 16 -> 1`, 3,457 parameters.
- `SharedHighResolutionPairModel`: shared member encoders followed by a
  configurable pair head. The standard 47-scalar/one-context form uses
  `33 -> 32 -> 16 -> 1` and has 4,321 parameters. The pair-sum/balance form
  uses 46 member scalars and three context values, giving
  `35 -> 32 -> 16 -> 1` and 4,369 parameters.

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

## Validation procedure

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

## Current result artifacts

`results/` contains compact tables and regenerable figures for the controlled
studies completed so far. This is an evolving, validation-only evidence bundle;
it is not a complete result set or a final model ranking. See
`results/README.md` for the available studies and their interpretation limits.

Completed evidence in this branch includes the deterministic mechanics, the
measured-pT-only diagnostic, the representation and high-resolution studies,
and the validation-only recovery, OR, pT-factorial and pT-context comparisons.
The latter adaptive studies have not passed protected confirmation. Planned
work includes confirmation, further controlled feature/model studies,
deployment assessment and a final unbiased evaluation. No blind-test result is
included or claimed.

## Tracked artifacts

The branch retains more than only the best current configuration. It keeps
reusable implementations and readable per-seed configurations for controlled
experiments that support the reported
conclusions, including scientifically useful negative results. This makes it
possible to reproduce why a direction was retained or rejected.

The repository intentionally excludes raw data, prepared caches, checkpoints,
pair/event prediction tables, temporary execution manifests and private
research notes. Compact result tables and regenerable figures are kept in
`results/` instead.

## Repository structure

```text
src/
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

The original single-object modules remain available because this branch starts
from the existing framework. Pair-specific code uses the same separation of
data preparation, features, models, training, calibration, evaluation and
plotting.

## Data

Place the externally supplied files under a data directory with this layout:

```text
Signal/signal_combined.csv
Signal/signal_combined.npz
Background/bkg_combined.csv
Background/bkg_combined.npz
```

Raw data, checkpoints, predictions and generated caches are intentionally not
tracked by Git. The `prongs` field is not used as a model input.

## Example workflow

Install the dependencies:

```bash
pip install -r requirements.txt
```

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

The tests cover same-event pairing, top-four selection, deterministic ties,
label construction, raw and engineered feature mappings, model dimensions,
maximum event aggregation, no-pair behavior, independent calibration,
cross-fitted validation and training-only normalization.

Some inherited single-object integration tests load pretrained experiment
weights. Those checks are skipped when the ignored external `experiments/`
artifacts are not present; their configuration and architecture unit tests
still run.

## Current scope

No binding hardware limit is applied to the research comparisons. The code
reports model input widths and parameter counts so deployment constraints can
be assessed when they are available.
