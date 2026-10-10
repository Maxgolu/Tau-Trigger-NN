# Single-object evidence package

This package preserves existing local experiment artifacts, not newly run
experiments. It complements the independent pair evidence elsewhere in `results/`.
The [reproduction guide](../../docs/SINGLE_OBJECT_REPRODUCTION.md) covers training.

## Contents and provenance

- `pretrained.zip`: all 78 distinct original checkpoints needed by the 114
  original FT configurations, with each parent run's `config.json`. Original
  relative `experiments/...` paths are retained. No raw detector data are included.
- `metrics.zip`: 586 unchanged `metrics.json` files and corresponding saved
  configs, compressed under `metrics/<original batch>/<original run>/`.
- `manifest.json`: individual artifact hashes, archive hashes, FT dependencies,
  included metric paths and explicit reasons for exclusions.
- `test_counts.json`: aggregate truth-tau bin counts for seeds 42/123/456,
  reconstructed from original CSV event metadata with the code's complete-event
  split. The four source data hashes and count method are recorded. No event
  identifiers or per-object detector inputs are distributed here.
- `benchmark_map.json`: exact main/progression/FT model-to-run/config mapping.
- `benchmark_summary.csv`: recomputed per-band efficiencies, global mean/SE
  and achieved FPR range, for the mapped three-seed models.

The archive selection requires the primary test-recalibrated operating point,
0.5% requested event FPR, 44 truth-pT bins from 10 to 120 GeV, and matching
test-count/binomial-error conventions. 178 older, differently calibrated or
different-population metric files were excluded rather than mixed silently.
Their paths and reasons remain in the manifest. The original research code
and configs remain in the repository; exclusion here is not deletion of work.

## Numerical reconstruction (Python standard library only)

From the repository root:

```text
python tools/single_object_artifacts.py verify
python tools/single_object_summary.py
python -m unittest discover -s tests -p test_single_object_reproduction.py -v
```

The summary pools fine bins using each seed's own truth-tau counts, then
averages efficiencies across seeds. Global efficiency is taken from each
saved measurement, not inferred from the five band means. Reported uncertainty
is sample standard deviation across three seeds divided by sqrt(3).
Inspect achieved FPR, not only the requested 0.5% value.

The five main global means reconstructed on packaging were 67.97, 69.61,
69.08, 68.40 and 67.47 percent, in paper-row order. All six generated LaTeX
result tables matched the existing submitted-source tables after line-ending
normalization. This checks arithmetic and provenance, not stochastic retraining.

## Recreate the six result figures and six tables

Use the plotting dependencies from the root requirements, then run:

```text
python tools/single_object_artifacts.py restore-metrics
python paper/figures/scripts/results/make_results.py main analysis ft
python paper/figures/scripts/results/lb.py
```

Output is `results/generated/single_object/paper/{figures,tables}/`.
The approved paper/PDF is **not overwritten**. PDF and PNG figures are produced.
Font availability can change typography; byte-identical PDFs are not promised.
No training, checkpoint loading, detector data or GPU is needed for these plots.

The recovered scripts have portable repository-relative defaults and no longer
depend on a private `obj.pkl`. A missing three-seed measurement is now an error,
not a fallback to report numbers. Optional environment overrides are
`TAU_EXPERIMENTS` (metric run tree), `TAU_COUNTS` (matching aggregate counts), and
`TAU_PAPER` (output folder). Do not use original counts with a different dataset.

Two explicitly historical panels remain historical: the prior-research fitted
curve panel and the `OR_PAIRS` reported-band comparison in the analysis figure.
Those are reconstructed from the original reporting constants, not newly
inferred predictions. The prior model's 62.53% is an author-confirmed supplied
measurement. The main single-object curves, progression and leaderboard rows
use the included per-seed metrics. Unknown/duplicate-seed leaderboard families
are not silently combined; two unsupported historical descriptors can produce
informational `no description` messages without affecting the six reproduced tables.

## FT artifact restoration

```text
python tools/single_object_artifacts.py restore-pretrained
```

This is opt-in: it restores verified parent weights/configs to the repository's
ignored `experiments/` tree. It refuses different existing files, validates paths,
and never deserializes Torch checkpoints. Checksums detect changes but do not
make an untrusted checkpoint safe; load weights only from a trusted source.
For fresh parent training, use `tools/bind_single_object_ft.py` as documented
in the reproduction guide. Do not rerun every historical experiment at once.
