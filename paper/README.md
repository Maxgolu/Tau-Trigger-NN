# Unified university report

This directory contains the complete editable Overleaf source combining Max
Golubkov's single-object work with Tamar Averbuch's pair-object contribution.
Max's existing source was preserved and the pair methodology, final result,
lessons, future direction, introduction, and reproducibility appendix were
integrated additively.

The compiled review PDF is `tau_trigger_unified_report.pdf`. The current full
scientific draft is 36 pages including references and appendices; the main text
ends on page 10. This is the complete-content draft. Page-limit editing remains
a separate review step and should move detail to appendices rather than remove
scientific qualifications.

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
