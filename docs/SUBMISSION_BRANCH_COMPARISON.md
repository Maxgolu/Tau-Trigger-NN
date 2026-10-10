# Submission Branch Comparison

## Purpose

This document records how the university submission branch was assembled and
which source is authoritative for each part of the final system.

## Branch ancestry

`university-submission` starts from local commit `424edf7` on
`pair-model-public`. That branch has `origin/main` commit `92a6edf` as its exact
merge base and adds twelve pair-model commits. It therefore preserves Max's
single-object history and the public pair-development history.

The remote `origin/pair-model-public` stops at `3445bbb`. Six later public
commits existed only locally when this submission branch was created.

## Authoritative sources

| Concern | Public branch state | Authoritative final evidence | Submission action |
|---|---|---|---|
| Single-object framework | Present | `origin/main` history | Preserve |
| Pair dataset and event aggregation | Present for top-four pairs | Final contract and one-shot runner | Extend without breaking top-four mode |
| Shared high-resolution pair model | Present | `src/phase8c_b01_train_only_core_v2.py` and frozen checkpoints | Reconcile and regression-test 4,357 parameters |
| Pair training | Equal-pair public runner | All-finite v4 runtime controls and training runner | Add equal-event reduction |
| Pair population | Top four by default | All finite TOBs, all unordered pairs | Add explicit configurable population |
| Training pair filter | Earlier eligible population | Same-label teaching pairs | Add explicit training-only filter |
| Inference population | Top-four pairs | Every finite pair; no truth filtering | Add and test all-finite inference |
| Event score | Maximum pair score | Maximum over complete finite-pair inventory | Preserve and test on all-finite mode |
| Final classifier | Public OR utilities exist | Frozen neural threshold OR deterministic pT threshold | Add final seed manifests |
| Final results | Validation summaries only | Final-test result and presentation handoff | Publish compact final summaries |
| Demo | Missing | Final system contract | Add CPU demo and expected output |
| Environment | Unpinned requirements; no pytest | Clean-clone requirement | Constrain versions and add test dependency |

## Frozen final-system evidence

The final system is bound by:

- `research_program/protected_confirmation_all_finite_equal_event_v1/final_test_contract_v1.json`;
- `research_program/protected_confirmation_all_finite_equal_event_v1/final_test_once_v1.py`;
- `research_program/reporting_staging/final_test_slide_handoff_v1/final_test_result.json`;
- `research_program/reporting_staging/final_test_slide_handoff_v1/PRESENTATION_HANDOFF.md`;
- the winner-freeze and output manifests referenced by those files.

The public implementation must reproduce their relevant behavior without
copying private paths, one-shot execution controls, internal audit machinery,
or protected raw data into the submission branch.

## Final behavior to port

1. Retain every finite-`tob_pt` member in deterministic order.
2. Construct every unordered pair.
3. Use same-label pairs for the selected training objective.
4. Average weighted pair BCE within each event, then average events.
5. Score every finite pair at inference without using truth labels.
6. Use the maximum pair logit as the neural event score.
7. OR the neural decision with the frozen deterministic pT branch.
8. Keep seeds 42, 123, and 456 separate and report their arithmetic mean and
   sample standard deviation.

## Public API changes required

### `src/pair_data.py`

Add an explicit `max_selected_tobs` option. The existing value `4` preserves
historical behavior; `None` selects all finite TOBs.

### `src/prepare_pair_data.py`

Add explicit `pair_population` and `training_pair_filter` configuration fields.
The pair population controls construction. The training filter applies only to
training arrays and must not affect validation, demo, or inference pair
enumeration.

### `src/pair_train.py`

Add complete-event batching and equal-event loss reduction. Preserve the
existing equal-pair mode for historical configurations.

### `src/pair_predict.py`

Use the pair inventory produced by the selected configuration. Do not inspect
truth labels when selecting inference pairs.

### Final manifests

Add one public final-system manifest containing architecture, preprocessing,
seed checkpoints, hashes, thresholds, aggregation, result-summary rules, and
the no-ensemble statement. Private absolute data bindings are excluded.

## Material intentionally not ported

The following remain in the internal research checkout or Git history:

- one-shot test execution controls;
- authorization and audit transaction machinery;
- private raw-data paths;
- large cached arrays and predictions;
- superseded phase-specific wrappers;
- unpublished checkpoints not needed by the demo;
- internal research identifiers that do not help reproduce the public method.

This is not deletion of Max's work. It is a boundary between the existing
public research framework and private experiment-governance artifacts.

## Verification target

The port is accepted only when tests prove:

- six finite TOBs create fifteen pairs in all-finite mode;
- top-four mode still creates six pairs;
- pair ordering is deterministic;
- equal-event reduction gives each event equal total contribution;
- inference pair selection is independent of truth labels;
- the selected model contains exactly 4,357 parameters;
- maximum-pair aggregation and neural-plus-pT decisions match saved fixtures.
