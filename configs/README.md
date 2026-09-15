# Pair experiment configuration map

Every JSON file defines one complete experiment. A folder may change the input
representation, neural architecture, training method or final event decision.
The folder name alone should not be interpreted as a distinct architecture.

## Current leading configurations

- [`pair_highres_two_channel_pt_or/`](pair_highres_two_channel_pt_or/) contains
  the current overall validation leader: absolute and signed-normalized EM2
  image channels with a jointly calibrated direct pT branch.
- [`pair_deeper_shared_encoder/`](pair_deeper_shared_encoder/) contains the
  controlled two-convolution encoder comparison.
- [`pair_highres_powerlaw_pminus1/`](pair_highres_powerlaw_pminus1/) contains
  the same learned architecture evaluated with a neural-only event decision.
- [`pair_rep_raw_shower_pt_context/`](pair_rep_raw_shower_pt_context/) contains
  the compact shared-MLP comparison.

Every study declares seeds 42, 123 and 456, either in one shared configuration
or in three seed-specific files.

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

The seed-42 scalar/coarse screen covered 22 exact candidates across these
families: raw coarse cells, per-layer summaries, compact EM2 measurements, EM2
core measurements, the EM2 best-window fraction, individual layer-removal
tests, and predefined detector-layer groups, with measured member pT included
only where declared. The exact candidate names and decisions are recorded in
[`../results/representation_reproduction/seed42_scalar_coarse_screen.csv`](../results/representation_reproduction/seed42_scalar_coarse_screen.csv).

The completed clean representation evidence is in
[`../results/clean_representation/summary.csv`](../results/clean_representation/summary.csv).

## Feature-source ablations

The `pair_source_ablation_*` folders remove one declared source at a time from
the current high-resolution input. These are meaningful negative experiments,
not alternative final models. Their compact result table is
[`../results/source_ablation/summary.csv`](../results/source_ablation/summary.csv).

## Output and fusion studies

- [`pair_output_ordered_threeclass/`](pair_output_ordered_threeclass/) changes
  the output from binary to operational classes 0, 1 and 2.
- [`pair_fusion_symmetric_threeclass/`](pair_fusion_symmetric_threeclass/)
  changes ordered member fusion to a symmetric sum-and-difference form.

## Event-context studies

- `pair_gain_*` isolates member pT, shower information and event context.
- `pair_context_*` compares seven alternative measured-pT descriptions: total
  event pT, highest pT outside the pair, total pT outside the pair, the pair's
  fraction of event pT, top-four pT concentration, top-four pT entropy, and
  pair sum plus balance. Top-four pT entropy is the Shannon entropy, in nats,
  of the four selected TOBs' measured-pT fractions. A larger value means a more
  even distribution; a smaller value means that pT is concentrated in fewer
  TOBs.
- `pair_highres_*_event_pt/` contains the high-resolution pT-factorial cells.

## Final-decision studies

- [`pair_raw_cell_or/`](pair_raw_cell_or/) tests a direct measured-pT branch
  with the raw-cell model.
- [`pair_highres_powerlaw_pt_or/`](pair_highres_powerlaw_pt_or/) records the
  earlier one-image-channel combined-decision study.

All tracked performance evidence is validation-only. See
[`../results/README.md`](../results/README.md) for result scope and limitations.
