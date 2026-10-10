# Main-paper page-limit edit plan

## Current pagination

- Max's exported PDF: Section 7 and the references begin on page 9; Appendix A
  begins on page 11.
- Unified PDF: Section 7 ends on page 10; references begin on page 11;
  Appendix A begins on page 13.
- The pair integration therefore adds approximately one main-text page. The
  original paper itself also requires roughly one page of compression to place
  all main sections within eight pages.

No page-limit edits below have been applied. They should be agreed with Max
before changing his existing prose.

## Recommended strategy

Preserve every result and qualification, but remove repeated technical detail
from the main text when the same material already appears in an appendix. The
main paper should retain the problem, method decisions, equations needed to
understand those decisions, headline comparisons, and limitations.

## Proposed reductions

| Priority | Main-text change | Appendix destination | Estimated saving |
|---|---|---|---:|
| 1 | Compress the four-item stabilisation discussion after the constrained-loss equation to one sentence | Appendix F already contains the full surrogate, cross-fitting, feasibility, and leakage-protection protocol | 0.45 page |
| 2 | Shorten the V1--V4 and E1--E3 architectural walkthrough to the design distinction and retained conclusion | Appendix G already explains every block and draws them | 0.30 page |
| 3 | Keep the pair construction, shared encoder, equal-event principle, and decision equation in Section 4.4, but move dimensions and training mechanics to Appendix K | Appendix K | 0.35 page |
| 4 | Keep the pair final table, 12.40-point gain, and failed success criterion in Section 5.3; move population accounting, regional-coordinate definition, and uncertainty details to Appendix K | Appendix K | 0.30 page |
| 5 | Merge the three pair-specific lesson paragraphs into one compact paragraph | Appendix K retains the technical rationale | 0.20 page |
| 6 | Tighten the introduction contribution list without removing either author's contribution | No move required | 0.15 page |

The estimated total is 1.75 pages. Float repacking may provide the remaining
space needed for Section 7 to finish by page 8.

## Content that must remain visible

- both authors and both the single-object and pair-object research programmes;
- the event-level 0.5% operating-point objective;
- the main single-object benchmark and its denominator;
- all-finite pair construction and equal-event training;
- the 4,357-parameter final pair architecture;
- 46.70% versus 34.30%, the 12.40-point gain, and 0.32-point seed spread;
- the 0.5183--0.5223% achieved FPR range;
- the statement that the predeclared conjunctive final criterion failed;
- the distinction between truth-tau-object and signal-event efficiencies;
- the no-ensemble and frozen-threshold protocol;
- the code and demo link.

## Review sequence

1. Max approves the proposed movements of his existing methodology text.
2. Apply all reductions in one revision, preserving the complete appendix text.
3. Recompile and confirm that references begin after eight pages of main text.
4. Review every page visually and compare all numerical claims again.
