# Result summaries

This directory contains the compact numeric evidence used by the report and by
`src/pair_plots.py`.
Raw detector data, checkpoints and event-level predictions are intentionally
not tracked.

`final_test/` is the submission-facing result release for the frozen
all-finite, equal-event candidate. It records the three selected seed systems,
their final test efficiencies and background false-positive rates, and the
cross-seed summary. The remaining directories preserve the controlled studies
that motivated and tested the final design.

The latest controlled evidence is summarized in:

- `final_test/final_result_summary.json`: compact final test result and scope;
- `final_test/seed_results.csv`: one auditable row per frozen seed system;

- `current_configurations.csv`: current leading configurations;
- `high_resolution_two_channel/summary.csv`: per-seed counts for the current
  two-channel leader;
- `source_ablation/summary.csv`: five removals from the complete input;
- `clean_representation/summary.csv`: four smaller input representations;
- `shared_encoder/summary.csv`: the compatible two-convolution comparison.
- `plots/latest_controlled_comparisons.png`: a presentation-ready view of the
  same recent comparisons.

“Inclusive efficiency” is the number of accepted validation signal-sample
events divided by all 7,307 validation signal-sample events. Each learned seed
has its own event threshold, calibrated at no more than 0.5% background-event
false-positive rate.

The raw-OR, high-resolution-OR, pT-factorial and pT-context groups are
**adaptive validation evidence**. They have not passed protected confirmation
and must not be read as final promoted-model or blind-test results. They are
preserved as the validation history that preceded the separately frozen final
evaluation in `final_test/`.

- `current_configurations.csv` is the compact entry point for the current
  reference configurations. It distinguishes input representation, neural
  architecture and final event decision.

- `data_summary.json` records the source filenames and CSV hashes, fixed sample
  and pair counts, input units, and the operational-label limitation used by
  the pair workflow.
- `pt_control/` records training loss, cross-fitted checkpoint selection and
  full-validation results for the fixed measured-pT-only diagnostic. These are
  validation results, not final unbiased performance estimates.
- `training_feature_analysis/` records training-only single-feature rankings
  and pairs of redundant candidate inputs. It was used to define later
  representation experiments; it did not select a model.
- `representation_study/` records validation comparisons between the
  deterministic baseline, the pT-only control and the later calorimeter
  representations. It also separates the controlled event-context and shower
  information effects and the output/fusion checks.
- `representation_reproduction/` records the completed three-seed reproduction
  of three descriptive representation candidates: raw coarse cells without
  member pT, raw coarse cells with member pT, and compact EM2 measurements with
  member pT. It includes global validation efficiency and conditional
  low/medium/high-energy differences. These rows remain validation-only; at
  that stage the test set was unopened, before the later frozen final
  evaluation summarized in `final_test/`. A compact seed-42 table also
  preserves the complete scalar/coarse screen: 22 exact candidates across raw
  cells, summaries, EM2 subsets, layer-removal tests and layer-group tests were
  tested; 19 were closed after screening, and the three named above were
  reproduced across all three seeds. Later high-resolution image studies are
  not part of that screen.
- `high_resolution/` records the shared high-resolution model's validation
  summary and a conditional matched-member pT-region diagnostic. The regional
  coordinate is not an inclusive two-distinct-generator-tau quantity.
- `high_energy_recovery/` records separate high-energy recovery experiments.
  The rows are alternatives, not a cumulative sequence of model changes.
- `pt_factorial/` records the controlled comparison of member measured pT and
  event second-highest measured pT, both separately and together.
- `pt_context/` records the completed validation comparison of seven alternative
  pT-context descriptions: total event pT, highest or total pT outside the pair,
  pair/event pT fraction, top-four concentration, top-four entropy, and pair
  sum plus balance. None improved the retained member-pT plus event
  second-highest-pT reference; pair sum and balance was the closest alternative.
  `validation_regions_by_seed.csv` supplies the compact per-seed and
  low/medium/high-energy evidence.
- `high_resolution_pt_or/` records the jointly calibrated high-resolution
  neural decision and direct measured-pT branch, including overlap counts and
  the incremental resource cost of the direct branch.
- `pair_or/` records the validation-only joint pair-network plus measured-pT
  classifier. It passed the predeclared point-estimate gate but remains pending
  protected confirmation; it is not presented as a final selected model.
- `plots/` contains figures regenerated directly from the CSV groups above.

Except for `final_test/`, efficiencies in these files are validation results,
not final performance estimates. Gain-source entries are controlled mean
differences, not proof that the corresponding information source is uniquely
causal. The high-resolution experiment changed representation and architecture
together and should be interpreted accordingly.

Run `python src/pair_plots.py --plot all` from the repository root to reproduce
the figures.
