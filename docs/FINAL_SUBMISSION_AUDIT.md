# Final Submission Readiness Audit

This document defines what “ready” means for the joint university submission. It separates machine-verifiable release gates from the few decisions that still require human sign-off.

## Release Gates

| Gate | Evidence | Verification | Current status |
|---|---|---|---|
| Code correctness | Unit and integration tests | `python -m pytest -q` | Passing: 294 tests |
| Clean execution | Pinned Python 3.11 dependencies | Run in a newly created virtual environment | Passing |
| Out-of-the-box demonstration | Data-free deterministic CPU example | `python demo/verify_demo.py` | Passing |
| Release integrity | SHA-256 bindings in the final-system manifest | `python submission/verify_submission.py` | Passing |
| Result arithmetic | Per-seed CSV and final JSON summary | Verifier recomputes the mean, sample SD and baseline | Passing |
| Report consistency | LaTeX table, prose, manifest and result artifacts | `python submission/verify_report.py` | Passing |
| Research traceability | Appendix K paths and preserved research presentation | Report verifier checks every cited repository path | Passing |
| Citation integrity | All LaTeX citation keys exist; pair directions cite primary work | Report verifier plus LaTeX compilation | Passing |
| Typesetting | Complete Overleaf source | Tectonic compilation with no errors or unresolved citations | Passing |
| Final caveat | FPR range and failed conjunctive rule beside the efficiency claim | Report verifier checks the exact qualification | Passing |

The authoritative single command is:

```bash
python submission/verify_submission.py
```

It verifies release-file hashes, recomputes the final summary, checks the report contract, and executes the demo. CI additionally runs the full test suite before this command.

## Numerical Contract

The final report, result artifacts and presentation crosswalk must agree on the following facts:

- Three separately trained seeds: 42, 123 and 456; no ensemble.
- Mean signal-event efficiency: 46.70%, compared with 34.30% for the deterministic baseline.
- Mean paired improvement: 12.40 percentage points; seed sample SD: 0.32 points.
- Achieved background-event FPR: 0.5183%--0.5223%.
- All three FPR values exceed the predeclared 0.5% ceiling, so the conjunctive final success criterion fails.
- Final population: 64,286 events, including 14,509 signal and 49,777 background events; 361,685 finite TOBs, 866,872 pairs and 589 no-pair events.
- Final model size: 4,357 trainable parameters.

The report verifier derives these values from `results/final_test/final_result_summary.json` and `results/final_test/seed_results.csv`; it does not treat duplicated prose as an independent source of truth.

## Max-Style Writing Contract

New prose should follow the terminology and argumentative style already established in Max's original report:

- Use British spelling: `normalised`, `normalisation`, `optimisation`, `labelled`, `unlabelled`, `programme` and `summarised`.
- Introduce the physics problem before the LHC rate and define each acronym before using it.
- Prefer the concrete terms already used by Max: `TOB`, measured `p_T`, event fake rate, `p_T` baseline, validation and final test.
- Call the extension the `pair model` or say that it `scores two TOBs together`; avoid introducing `pair-object` as a second name.
- Say that `every event receives equal weight` rather than naming an `equal-event reduction`.
- Name the metric before quoting it. The pair result counts the fraction of signal events that fire; the earlier study counts accepted truth-tau objects. They are not directly comparable.
- Separate selection from evaluation: checkpoints and thresholds are selected on validation, then applied unchanged to the final test.
- Put qualifications beside the claim they limit. The 46.70% result must remain adjacent to the 0.5183%--0.5223% FPR range and the failed success rule.
- Prefer restrained claims such as “improves the point estimate”, “is consistent with”, “does not isolate” and “cannot be proven”; do not imply causality from the final comparison alone.
- Use compact paragraph leads to state the finding, then give the protocol, evidence and limitation. Move implementation detail to an appendix and cross-reference it.
- Report the three seeds separately and summarise them arithmetically; never describe their mean as an ensemble prediction.

Avoid internal experiment labels in the main text when an ordinary explanation
is available. In particular, do not use `frozen candidate`, `threshold-transfer
test`, `operational class`, `member encoder`, `pair population` or `conjunctive
success rule`. The appendix may give exact implementation rules, but it should
still explain them using TOBs, events, labels, thresholds and the final test.

`submission/verify_report.py` automatically enforces the most important vocabulary, metric-definition and caveat rules. Final human proofreading remains necessary for tone and readability because those properties cannot be fully reduced to pattern matching.

## Manual Sign-Off

Before delivery, both authors should confirm:

1. Max approves any shortening or movement of his original prose needed to meet the eight-page main-text limit.
2. The lecturer's expected deliverables are exactly the repository link, executable demo and unified report, with no separate upload format or institutional template.
3. A person other than the implementer follows the README from a clean clone on a CPU-only machine.
4. The final PDF is visually inspected after the last edit, especially tables, references, figure placement and appendix links.
5. The release commit and optional tag are created only after all automated gates pass; the manifest is then updated from `release_candidate_uncommitted` to the final released state.

## Scope Discipline

The submission keeps Max's work and the research history. It does not present every historical experiment as part of the final executable path. The README and manifest identify one authoritative final pair configuration, one compact result package and one demo. Historical experiments remain available as research evidence and are explicitly distinguished from the frozen final evaluation.
