# Result summaries

This directory contains the compact numeric inputs used by `src/pair_plots.py`.
Raw detector data, checkpoints and event-level predictions are intentionally
not tracked.

The bundle contains only the controlled studies completed so far. It will be
extended as the research programme continues; it is not a comprehensive final
result release.

The raw-OR, high-resolution-OR, pT-factorial and pT-context groups are
**adaptive validation evidence**. They have not passed protected confirmation
and must not be read as final promoted-model or blind-test results. Protected
confirmation and the final unbiased evaluation remain planned work.

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
- `high_resolution/` records the shared high-resolution model's validation
  summary and a conditional matched-member pT-region diagnostic. The regional
  coordinate is not an inclusive two-distinct-generator-tau quantity.
- `high_energy_recovery/` records separate high-energy recovery experiments.
  The rows are alternatives, not a cumulative sequence of model changes.
- `pt_factorial/` records the controlled comparison of member measured pT and
  event second-highest measured pT, both separately and together.
- `pt_context/` records the completed validation comparison of seven alternative
  pT-context descriptions. None improved the retained member-pT plus event
  second-highest-pT reference; pair sum and balance was the closest alternative.
  `validation_regions_by_seed.csv` supplies the compact per-seed and
  low/medium/high-energy evidence used by the presentation notes.
- `high_resolution_pt_or/` records the jointly calibrated high-resolution
  neural decision and direct measured-pT branch, including overlap counts and
  the incremental resource cost of the direct branch.
- `pair_or/` records the validation-only joint pair-network plus measured-pT
  classifier. It passed the predeclared point-estimate gate but remains pending
  protected confirmation; it is not presented as a final selected model.
- `plots/` contains figures regenerated directly from the CSV groups above.

All efficiencies in these files are validation results. They are not final
unbiased performance estimates. Gain-source entries are controlled mean
differences, not proof that the corresponding information source is uniquely
causal. The high-resolution experiment changed representation and architecture
together and should be interpreted accordingly.

Run `python src/pair_plots.py --plot all` from the repository root to reproduce
the figures.
