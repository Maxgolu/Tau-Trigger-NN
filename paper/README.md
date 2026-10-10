# Unified university report

This directory contains the complete editable Overleaf source combining Max
Golubkov's single-object work with Tamar Averbuch's pair-object contribution.
Max's existing source was preserved and the pair methodology, final result,
lessons, future direction, introduction, and reproducibility appendix were
integrated additively.

The compiled review PDF is `tau_trigger_unified_report.pdf`. The final report
is 37 pages including references and appendices. The main text ends on page 8,
references begin on page 9 and appendices begin on page 10. Detailed material
was moved to appendices without removing scientific qualifications.

Overleaf build sequence:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The final pair result reports 46.70% mean signal-event efficiency versus 34.30%
for the deterministic baseline, together with the required qualification that
all three final background FPR values exceed the predeclared 0.5% limit.

The agreed October 10 revision restores the original Section 7 and Appendix
J.3 passages before adding pair-study extensions. The original scientific
source, figures and tables are checked by `submission/verify_report.py` using
`submission/original_report_preservation.json`. The six methodological
clarifications are integrated, and the fake-rate qualification occupies one
sentence beside the pair result.

Appendix K.3 (Table 18, page 36) preserves six concrete historical validation
findings and the cross-layer spatial-compatibility caveat. The results are
labelled adaptive validation, distinct comparison settings are retained, and
the added numerical checks recompute the table from public aggregate files.
