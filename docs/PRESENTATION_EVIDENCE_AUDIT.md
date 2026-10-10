# Pair Presentation Evidence Audit

## Purpose

The preserved 47-slide pair-model presentation
`presentation/pair_model_research_history.pptx` documents the research path in
substantially more detail than the final joint presentation. Its speaker notes
also record the intended interpretation and public code/result references for
most experiments. This material should support the unified report and release
audit; it should not be mistaken for the final numerical contract.

## Evidence hierarchy

When two artifacts describe different stages, use this order:

1. `submission/final_system_manifest.json` for the frozen submitted system;
2. `results/final_test/final_result_summary.json` and `seed_results.csv` for
   final-test numbers;
3. `atlas-presentation.pptx` for the concise joint presentation narrative;
4. the 47-slide pair presentation and its notes for method history, controlled
   validation experiments, and provenance;
5. older result folders for supporting experiment-level detail.

This order preserves the full research record without allowing an earlier
validation value or superseded top-four method to replace the final result.

## Content map

| Slides | Contribution to the unified record | Status |
|---|---|---|
| 1--10 | Pair question, all-finite construction, deterministic baseline, maximum aggregation, and fair independent thresholding | Reuse as method evidence |
| 11--17 | Data/evaluation machinery and the measured-pT-only diagnostic | Historical control study |
| 18--32 | Representation ladder, broad input screen, gain-source study, and output/fusion comparisons | Summarize selectively; details belong in an appendix |
| 33--39 | High-resolution shared encoder and neural-plus-measured-pT event decision | Reuse as final architecture evidence |
| 40--44 | Controlled measured-pT, context, and shower-width studies | Supporting ablations; do not present as final results |
| 45 | Pre-final freeze of the all-finite, equal-event candidate | Reuse as protocol evidence, not as final numerical evidence |
| 46 | Symmetric-fusion and whole-pair alternatives | Supporting architecture ablation |
| 47 | Hardware-assessment boundary | Reuse in limitations and future work |

## Speaker-note evidence retained

The notes explain several details that the compact final deck omits:

- six finite TOBs produce fifteen unordered pairs;
- exact measured-pT ties use `tob_index` only for deterministic member order;
- same-label pairs define the selected teaching loss, while inference scores
  every finite pair without truth information;
- loss is averaged within each event before events are averaged equally;
- events with no pair receive a non-passing neural score but stay in applicable
  event denominators;
- the same learned member encoder processes both TOBs;
- the maximum pair score is the neural event score;
- the neural and deterministic measured-pT branches share one background
  budget and are combined by event-level OR;
- seeds 42, 123, and 456 are separate systems rather than an ensemble;
- checkpoint selection, preprocessing, thresholds, and hashes were frozen
  before final evaluation;
- unique generator-tau identity is unavailable, so operational labels must not
  be described as proving two distinct generator taus;
- hardware feasibility remains a later engineering assessment rather than a
  demonstrated deployment result.

These points belong in the main methodology or technical appendix of the
unified report, depending on space.

## Historical statements that must stay labelled historical

The following claims describe earlier validation stages and must not be copied
into the final-results section without their stage labels:

- the top-four method and its maximum of six pairs;
- the measured-pT-only diagnostic and its selected validation epochs;
- the 42.46%, 44.57%, 44.76%, and 47.24% validation efficiencies;
- the 0.4993% validation FPR;
- statements that the protected test set remained unopened;
- candidate-specific comparisons for raw cells, context alternatives,
  shower width, symmetric fusion, and the whole-pair MLP.

They remain useful evidence of a controlled research programme and explain why
the final architecture and training population were selected.

## Final-result replacement

The final report and submission must instead state:

- mean final-test signal-event efficiency: 46.70%;
- deterministic measured-pT baseline: 34.30%;
- mean improvement: 12.40 percentage points;
- sample standard deviation across independently trained models: 0.32
  percentage points;
- final background FPR range: 0.5183--0.5223%;
- all three final systems exceeded the predeclared 0.5% background-FPR limit,
  so the conjunctive final success criterion failed.

The efficiency gain remains a valid descriptive final-test observation, but it
must be reported together with the operating-point qualification.

## Final branch references

The submission-facing evidence for the retained claims is:

- `src/pair_data.py`: all-finite construction and deterministic pair order;
- `src/prepare_pair_data.py`: training-only same-label filtering;
- `src/pair_train.py`: event-balanced loss, seeded initialization, and frozen
  normalization;
- `src/pair_models.py`: shared member encoder and 4,357-parameter model;
- `src/pair_classifiers.py`: maximum aggregation and event-level OR decision;
- `configs/pair_final_all_finite_equal_event/config.json`: frozen final method;
- `results/final_test/final_result_summary.json`: final aggregate result;
- `results/final_test/seed_results.csv`: separate seed outcomes;
- `submission/final_system_manifest.json`: hashes, thresholds, provenance, and
  final-criterion status;
- `demo/run_demo.py` and `demo/verify_demo.py`: executable functional example;
- `tests/`: regression coverage for the final contract.

## Submission decision

Do not delete the detailed presentation or the experiments it describes. Keep
them as research-history evidence. The README, demo, frozen config, result
manifest, and unified report should point readers first to the selected final
workflow, while appendices and this audit explain how the project reached it.
