# Single-object submission handoff - 2026-10-11

Local packaging update to `university-submission` at base commit `3ea740e`.
No training, neural-source changes or original-config edits were made.

## Changes

- Root README links a complete single-object run/reproduction guide and the
  preserved original research README.
- `results/single_object/` contains two compact archives (about 12 MB total):
  78 FT parent weights with configs; 586 original per-seed metric files with
  configs. Checksums, aggregate test counts and the paper mapping are included.
- `tools/` verifies/restores artifacts, recomputes benchmark summaries, and
  safely binds a newly trained parent to a new FT config.
- `paper/figures/scripts/results/` contains the restored portable plotting and
  table scripts. Generated products are isolated under ignored `results/generated/`.
- The approved author-supplied PDF replaces `paper/tau_trigger_unified_report.pdf`.
  Its existing links remain valid; the PDF/README release bindings were updated.

## Verification performed

- 114 FT references resolve to archived weights; parent seeds, features and
  architectures match the target FT configurations.
- All five main benchmark rows and all six result tables reconstruct the
  existing paper-source values; all six figures regenerated and visually checked.
- Six data-free packaging/reproduction unit tests passed. Plot checks used
  available Python 3.12/NumPy with an isolated Matplotlib 3.9.4 installation,
  not a newly installed full pinned training environment.
- No model was trained or run on detector data. Archived metrics are historical
  evidence, not a new independent replication.

## Remaining separate handoff items

1. The final PDF is approved, but the existing LaTeX is an earlier source snapshot.
   Add the matching final Overleaf source before claiming source/PDF equivalence.
2. Both studies need approved detector data. The five frozen external normalizer
   files in the pair final config are a pair-specific delivery requirement.
3. The pre-existing main submission verifier has Windows CRLF and demo-path
   issues. These are unchanged by this scoped update. The new single-object
   verifier normalizes only its text metadata hashes; binary archives and
   checkpoint hashes remain byte-exact.
4. These changes are local until reviewed and committed/pushed. No remote
   branch or release was updated by this task.
