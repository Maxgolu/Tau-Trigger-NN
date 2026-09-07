# Pair experiment configuration map

Every JSON file defines one complete experiment. A folder may change the input
representation, neural architecture, training method or final event decision.
The folder name alone should not be interpreted as a distinct architecture.

## Current leading configurations

- [`pair_highres_powerlaw_pt_or/`](pair_highres_powerlaw_pt_or/) contains the
  current overall validation leader: the shared high-resolution network with a
  jointly calibrated direct pT branch.
- [`pair_highres_powerlaw_pminus1/`](pair_highres_powerlaw_pminus1/) contains
  the same learned architecture evaluated with a neural-only event decision.
- [`pair_rep_raw_shower_pt_context/`](pair_rep_raw_shower_pt_context/) contains
  the compact shared-MLP comparison.

Each folder contains separate configurations for seeds 42, 123 and 456.

## Diagnostic controls

- [`pair_pt_control/`](pair_pt_control/) validates the learned pair pipeline
  using only the two measured member-pT values.
- [`pair_raw_cells/`](pair_raw_cells/) provides the flat raw-cell control.

These controls are not current leading models.

## Input-representation studies

- `pair_rep_*` contains the first controlled representation study.
- `pair_coarse_cells_*` and `pair_compact_em2_pt_shared/` contain the later
  three-seed representation reproduction.
- `pair_high_resolution_em2/` and `pair_highres_*` contain the high-resolution
  EM2 studies and their controlled input variants.

## Output and fusion studies

- [`pair_output_ordered_threeclass/`](pair_output_ordered_threeclass/) changes
  the output from binary to operational classes 0, 1 and 2.
- [`pair_fusion_symmetric_threeclass/`](pair_fusion_symmetric_threeclass/)
  changes ordered member fusion to a symmetric sum-and-difference form.

## Event-context studies

- `pair_gain_*` isolates member pT, shower information and event context.
- `pair_context_*` compares seven alternative measured-pT context summaries.
- `pair_highres_*_event_pt/` contains the high-resolution pT-factorial cells.

## Final-decision studies

- [`pair_raw_cell_or/`](pair_raw_cell_or/) tests a direct measured-pT branch
  with the raw-cell model.
- [`pair_highres_powerlaw_pt_or/`](pair_highres_powerlaw_pt_or/) applies the
  same type of combined event decision to the current high-resolution leader.

All tracked performance evidence is validation-only. See
[`../results/README.md`](../results/README.md) for result scope and limitations.
