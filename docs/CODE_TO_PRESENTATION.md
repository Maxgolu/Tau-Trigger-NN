# Code, Report, and Presentation Traceability

## Authority

The submitted code and report use the frozen final-system manifest as the
authoritative technical contract. The concise joint presentation communicates
the selected system and headline result. The preserved 47-slide pair-study
deck supplies the detailed research path, speaker-note explanations, and
historical experiment references.

The final presentation has already been delivered and does not need to be
rewritten. Where its compact result slide omits a qualification, the unified
report and repository evidence provide the complete statement.

## Final-system crosswalk

| Claim | Final joint presentation | Detailed pair deck | Code and tests | Report insertion | Evidence status |
|---|---|---|---|---|---|
| Keep every finite TOB and form every unordered same-event pair | Slide 13 | Slides 3, 6, and 7 with notes | `src/pair_data.py`; `tests/test_pair_data.py` | Section 4.4, “From independent objects to same-event pairs”; technical appendix | Verified |
| Six finite TOBs produce fifteen pairs | Slide 13 implies all pairs | Slides 3, 6, and 7 | Demo regression and `tests/test_pair_data.py` | Section 4.4 and appendix | Verified |
| Higher-pT member is first; `tob_index` breaks ties | Slide 13 | Slides 3 and 7 notes | `src/pair_data.py`; deterministic-order tests | Section 4.4 and appendix | Verified |
| Both members use the same encoder | Slides 10, 11, and 13 | Slides 33, 34, and 39 | `src/pair_models.py`; architecture tests | Section 4.4, “A shared member encoder” | Verified |
| Final network has 4,357 parameters | Slides 12 and 13 | Slides 34, 39, and 45 | Parameter-count regression and demo | Section 4.4 and final-results text | Verified |
| Training removes mixed-label teaching pairs only | Slide 13 does not detail labels | Slide 7 notes | `src/prepare_pair_data.py`; preparation tests | Section 4.4 and appendix | Verified |
| Truth is never required at inference | Not stated explicitly | Slides 7 and 39 notes | Pair-preparation and inference tests | Section 4.4 and appendix | Verified |
| Every event has equal total training influence | Slide 9 | Slides 7 and 45 notes | `src/pair_train.py`; equal-event-loss tests | Section 4.4, “Training pairs and equal-event reduction” | Verified |
| Neural event score is the maximum pair score | Slides 7 and 13 | Slides 9, 34, and 39 | `src/pair_data.py`; aggregation tests; demo | Section 4.4, “From pair scores to one trigger decision” | Verified |
| Neural and measured-pT branches are combined by event-level OR | Slides 7, 8, and 13 | Slides 37 and 39 | `src/pair_classifiers.py`; classifier tests; demo | Section 4.4 decision equation | Verified |
| Seeds 42, 123, and 456 are independent systems, not an ensemble | Slide 16 | Slide 45 and notes | Final config, seed CSV, and manifest | Section 4.4 and result table | Verified |
| Frozen epochs are 17, 18, and 20 | Not shown | Slide 45 and notes | Final config and manifest | Pair-object results and appendix | Verified |
| Final signal-event efficiency is 46.70% versus 34.30% | Slide 16 | Earlier deck stops at the frozen validation candidate | `results/final_test/final_result_summary.json`; `seed_results.csv`; submission verifier | Pair-object final-evaluation table | Verified |
| Mean gain is 12.40 percentage points with 0.32-point seed SD | Slide 16 | Earlier deck stops at the frozen validation candidate | Final result files and verifier recomputation | Pair-object final-evaluation table and text | Verified |
| Final FPRs are 0.5183--0.5223%, above the 0.5% ceiling | Omitted from Slide 16 | Slide 45 records only the pre-final validation operating point | Final result files and manifest | “Protocol qualification” paragraph | Verified in repository and report draft |
| The predeclared conjunctive final success criterion failed | Omitted from the delivered deck | Not available before final evaluation | `submission/final_system_manifest.json` | “Protocol qualification” paragraph | Verified in repository and report draft |
| Demo runs without private data or checkpoints | Not a scientific slide claim | N/A | `demo/run_demo.py`; `demo/verify_demo.py`; clean-environment verification | Reproducibility appendix may cite repository | Verified |

## Historical experiments

Earlier top-four, representation, context, fusion, and shower-width studies
remain in the repository and in the detailed pair deck. Their validation
results explain model selection but do not replace the final all-finite result.
The report should summarize the strongest decisions in the main text and move
the detailed experiment sequence to an appendix if space permits.

The key historical transition is:

1. validate pair construction and event reconstruction with a measured-pT-only
   control;
2. compare richer calorimeter representations under a fixed evaluation rule;
3. retain the high-resolution shared-member architecture;
4. add the joint neural and deterministic measured-pT event decision;
5. test context, feature, and fusion alternatives without promoting variants
   that did not improve validation;
6. freeze the all-finite, equal-event candidate before final evaluation.

## Release checks

Before release, this matrix must be reviewed against the integrated Overleaf
source, not only the standalone insertion draft. The final audit must confirm:

- every retained report number comes from `results/final_test/` or a clearly
  labelled validation artifact;
- the report distinguishes signal-event efficiency from truth-tau-object
  efficiency;
- the final operating-point qualification remains visible;
- the introduction reflects the completed joint paper and is written last;
- no historical validation result is described as a final-test result.
