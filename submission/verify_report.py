"""Verify the unified report's numerical, evidential, and terminology contract."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SUMMARY = ROOT / "results" / "final_test" / "final_result_summary.json"
SEED_RESULTS = ROOT / "results" / "final_test" / "seed_results.csv"

REQUIRED_REPORT_FILES = (
    "main.tex",
    "sections/01_introduction.tex",
    "sections/04_methodology.tex",
    "sections/05_results.tex",
    "sections/06_lessons.tex",
    "sections/07_future.tex",
    "appendix/K_pair_object.tex",
    "tables/tab_pair_final.tex",
    "references.bib",
)

PAIR_EVIDENCE_PATHS = (
    "configs/pair_pt_control",
    "results/pt_control",
    "results/representation_study",
    "results/representation_reproduction",
    "results/high_resolution",
    "results/high_resolution_two_channel",
    "results/pair_or",
    "results/high_resolution_pt_or",
    "results/pt_factorial",
    "results/high_energy_recovery",
    "results/pt_context",
    "results/source_ablation",
    "results/clean_representation",
    "results/shared_encoder",
    "configs/pair_final_all_finite_equal_event",
    "results/final_test",
    "presentation/pair_model_research_history.pptx",
)

PAIR_PROSE_FILES = (
    "sections/01_introduction.tex",
    "sections/04_methodology.tex",
    "sections/05_results.tex",
    "sections/06_lessons.tex",
    "sections/07_future.tex",
    "appendix/J_future.tex",
    "appendix/K_pair_object.tex",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(relative: str) -> str:
    return (PAPER / relative).read_text(encoding="utf-8")


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", strip_comments(text)).strip()


def verify_structure() -> None:
    for relative in REQUIRED_REPORT_FILES:
        require((PAPER / relative).is_file(), f"missing report file: paper/{relative}")

    main = strip_comments(read("main.tex"))
    require("Max Golubkov" in main and "Tamar Averbuch" in main, "joint authorship missing")
    require(
        r"\input{appendix/K_pair_object}" in main,
        "pair-object appendix is not included by paper/main.tex",
    )

    for path in PAPER.rglob("*.tex"):
        if path.name == "macros.tex":
            continue
        require(
            re.search(r"\\todo(?:\[[^]]*\])?\{", strip_comments(path.read_text(encoding="utf-8")))
            is None,
            f"active TODO remains in {path.relative_to(ROOT)}",
        )


def verify_citations() -> None:
    sources = "\n".join(
        strip_comments(path.read_text(encoding="utf-8")) for path in PAPER.rglob("*.tex")
    )
    cited: set[str] = set()
    for citation in re.findall(r"\\cite\w*\{([^}]+)\}", sources):
        cited.update(key.strip() for key in citation.split(","))
    bibliography = read("references.bib")
    defined = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography))
    missing = sorted(cited - defined)
    require(not missing, f"undefined bibliography keys: {', '.join(missing)}")
    require(
        {"zaheer2017deepsets", "lee2019settransformer"}.issubset(cited),
        "pair research directions must cite Deep Sets and Set Transformer",
    )


def verify_evidence_paths() -> None:
    appendix = read("appendix/K_pair_object.tex")
    for relative in PAIR_EVIDENCE_PATHS:
        require((ROOT / relative).exists(), f"missing pair evidence: {relative}")
        require(
            rf"\path{{{relative}" in appendix,
            f"pair evidence is not referenced in Appendix K: {relative}",
        )


def verify_numerical_contract() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    with SEED_RESULTS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    table = read("tables/tab_pair_final.tex")
    results = read("sections/05_results.tex")
    appendix = read("appendix/K_pair_object.tex")
    methodology = read("sections/04_methodology.tex")

    require(len(rows) == 3, "the final pair table must contain three seed systems")
    for row in rows:
        seed = row["seed"]
        expected_cells = (
            f"{float(row['candidate_efficiency']) * 100:.2f}",
            f"{float(row['baseline_efficiency']) * 100:.2f}",
            f"{float(row['paired_improvement']) * 100:.2f}",
            f"{float(row['background_fpr']) * 100:.4f}",
        )
        matching_line = next((line for line in table.splitlines() if line.strip().startswith(seed)), "")
        require(matching_line, f"seed {seed} is missing from the final report table")
        for value in expected_cells:
            require(value in matching_line, f"seed {seed} report value drift: {value}")

    efficiency = summary["inclusive_efficiency"]
    expected_summary_values = (
        f"{efficiency['candidate_seed_mean'] * 100:.2f}",
        f"{efficiency['candidate_seed_sample_sd'] * 100:.2f}",
        f"{efficiency['deterministic_baseline'] * 100:.2f}",
        f"{efficiency['mean_paired_improvement_percentage_points']:.2f}",
    )
    mean_line = next((line for line in table.splitlines() if line.strip().startswith("Mean")), "")
    for value in expected_summary_values:
        require(value in mean_line, f"mean report value drift: {value}")

    population = summary["test_population"]
    for value in population.values():
        tex_value = f"{value:,}".replace(",", "{,}") if value >= 1000 else str(value)
        require(tex_value in results or tex_value in appendix, f"test population value missing: {value}")
    parameter_text = f"{summary['model_parameters']:,}".replace(",", "{,}")
    require(parameter_text in methodology, "trainable parameter count drift")

    fprs = list(summary["background_fpr_by_seed"].values())
    fpr_range = f"{min(fprs) * 100:.4f}$--${max(fprs) * 100:.4f}"
    require(fpr_range in results, "background FPR range drift")
    require("required $0.5\\%$" in results, "the event fake-rate requirement is missing")
    require(
        "final success criterion" in results and "it is not met" in results,
        "the failed final success criterion is not stated explicitly",
    )


def verify_terminology_contract() -> None:
    prose = "\n".join(strip_comments(read(relative)) for relative in PAIR_PROSE_FILES)
    results = compact(read("sections/05_results.tex"))
    methodology = compact(read("sections/04_methodology.tex"))
    appendix = compact(read("appendix/K_pair_object.tex"))

    for spelling in ("normalized", "normalization", "optimization", "labeled", "unlabeled"):
        require(
            re.search(rf"\b{spelling}\b", prose, flags=re.IGNORECASE) is None,
            f"American spelling conflicts with Max's terminology: {spelling}",
        )

    discouraged_phrases = (
        "pair-object",
        "equal-event reduction",
        "frozen threshold-transfer test",
        "predeclared conjunctive",
        "operational class",
        "operational label",
        "member encoder",
        "final test population",
    )
    for phrase in discouraged_phrases:
        require(phrase not in prose.lower(), f"unnecessarily abstract report terminology: {phrase}")

    introduction = compact(read("sections/01_introduction.tex"))
    require(
        "Identifying tau particles is important" in introduction,
        "the introduction no longer begins from the reader-facing physics problem",
    )
    require("Trigger Objects (TOBs)" in introduction, "TOB is used before being introduced")
    require("signal efficiency is the fraction of signal events" in results, "pair event metric is not defined")
    require(
        r"fraction of truth-$\tau$ objects" in results,
        "pair and single-object efficiency definitions are not distinguished",
    )
    require("selected on validation" in results, "validation selection is not stated")
    require("then applied unchanged" in results, "unchanged test application is not stated")
    require("not combined into an ensemble" in methodology, "seed systems could be misread as an ensemble")
    require("not used for" in appendix and "recalibration" in appendix, "test-recalibration prohibition is missing")


def main() -> int:
    verify_structure()
    verify_citations()
    verify_evidence_paths()
    verify_numerical_contract()
    verify_terminology_contract()
    print("Report contract verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
