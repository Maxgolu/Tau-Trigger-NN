# Agreed report and training corrections, 10 October 2026

The original Overleaf export was located through the “Create unified report
and code” chat. Its source is the preservation reference; its SHA-256 is
recorded in `submission/original_report_preservation.json`.

- Restored Max's Section 4.4 headings, original Section 7 bullet and complete
  Appendix J.3 passage. Pair-study extensions follow the original passages.
- Preserved original scientific content in order and bound untouched methods,
  tables, figures and other source files by SHA-256. Only unfinished introduction
  and pair placeholders are replaced. Bibliography entries are preserved.
- Kept one sentence beside the pair results explaining the achieved fake rates
  and the unmatched-rate comparison; added the baseline fake rate to the caption.
- Distinguished retained training pairs and eligible training events from all
  inference pairs, retaining the existing signal-dependent weights.
- Specified the first eligible top-four pair for regional evaluation and added
  regional denominators in Appendix K.
- Made the common split seed 42 explicit, separate from the three training seeds.
- Marked the paired interval conditional on the observed discordant count.
- Corrected final all-finite/same-label preparation to retain negative pairs
  from signal events without two positive members. Historical preparation modes
  retain their prior event gate. Validation and inference retain all finite pairs.
- Regression checks exhaust all binary label patterns for zero through six TOBs
  and check mixed-label validation pairs.

Saved final-test decisions, selected checkpoints and their thresholds are
unchanged. No retraining or test-threshold recalibration was performed.
The new baseline and regional aggregate fields are recomputed from the saved
final-test decisions.

This is a local revision candidate after the existing release. No existing tag
was moved, and the revised source has not been committed or pushed. The
full-content PDF is now 37 pages; page-limit editing is separate.

## Approved historical-validation addition

Added one page in Appendix K.3, containing Table 18 and the cross-layer
cell-summation qualification. It preserves representation screening, the
shower/context interaction, output and fusion comparisons, member/event pT
inputs, unsuccessful context and width changes, topology tests and the
combined all-finite/event-averaged comparison. Distinct 44.756% and 44.569%
references are explicitly identified; these historical adaptive-validation
results are not independent confirmation or final-test results. The two
changes in the final comparison were evaluated together.

Numbers were checked against saved CSV/JSON results. Additional public
aggregate evidence and source hashes are in `results/historical_validation/`;
no event identities or raw data were added. The representation-screen pT
difference is rounded from saved results to 0.078 percentage points.

Coordinated ownership with “Create unified report and code”: that chat paused
report, manifest and release writes during this addition. Its intended v2
code link is preserved in both source copies. It retains ownership of release
metadata, commits, tagging, pushing and remote-clone verification.

The 37-page PDF compiles without undefined references or overfull boxes.
The cover and all Appendix K pages were visually checked after the final
build. Original-source preservation and numerical report verification pass.

Final checks: 296 tests passed (11 upstream deprecation warnings); report,
submission, demo, ZIP integrity, source synchronization and whitespace checks
passed. The previous Appendix K content is retained in its original order.
