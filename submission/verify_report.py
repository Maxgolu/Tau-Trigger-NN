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
    "results/historical_validation",
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
    require(fpr_range in appendix, "background FPR range drift")
    require("$0.5\\%$ fake-rate limit" in compact(appendix), "the event fake-rate requirement is missing")
    require(
        "final success criterion" in appendix and "It is not met" in appendix,
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
    # Retain Max's original titles even where newer prose uses "pair model".
    prose = prose.replace("Input unit: single-object and pair-object models", "")
    prose = prose.replace("Pair-object models.", "")
    prose = prose.replace("team's pair-object work", "")
    for phrase in discouraged_phrases:
        require(phrase not in prose.lower(), f"unnecessarily abstract report terminology: {phrase}")

    introduction = compact(read("sections/01_introduction.tex"))
    require(
        introduction.startswith("\\section{Introduction} \\label{sec:intro} The ATLAS Level-1 trigger must distinguish narrow tau deposits"),
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


def verify_original_preservation() -> None:
    contract = json.loads((ROOT / "submission" / "original_report_preservation.json").read_text())
    import hashlib

    def require_blocks_in_order(relative: str, blocks: list[str], origin: str) -> None:
        tokens = compact(read(relative)).split()
        cursor = 0
        for block in blocks:
            expected = block
            for replacement in contract.get("preserved_block_replacements", {}).get(relative, []):
                expected = expected.replace(replacement["original"], replacement["revised"])
            for token in expected.split():
                try:
                    cursor = tokens.index(token, cursor) + 1
                except ValueError:
                    raise RuntimeError(
                        f"Max's original content is missing or replaced: {origin} -> {relative}: {token}"
                    )

    for relative, expected in contract["unchanged_files"].items():
        require(hashlib.sha256((PAPER / relative).read_bytes()).hexdigest() == expected,
                f"Max's original file changed: {relative}")

    for relative, binding in contract.get("appended_files", {}).items():
        data = (PAPER / relative).read_bytes()
        original_length = binding["original_byte_length"]
        require(len(data) >= original_length, f"Max's original file was truncated: {relative}")
        require(
            hashlib.sha256(data[:original_length]).hexdigest() == binding["original_sha256"],
            f"Max's original file prefix changed: {relative}",
        )

    for relative, binding in contract.get("approved_replacements", {}).items():
        reconstructed = (PAPER / relative).read_text(encoding="utf-8")
        for replacement in binding["replacements"]:
            revised = replacement["revised"]
            require(
                reconstructed.count(revised) == 1,
                f"approved report replacement is missing or ambiguous: {relative}: {revised}",
            )
            reconstructed = reconstructed.replace(revised, replacement["original"], 1)
        require(
            hashlib.sha256(reconstructed.encode("utf-8")).hexdigest() == binding["original_sha256"],
            f"Max's original file changed beyond approved replacements: {relative}",
        )

    relocations_by_source: dict[str, list[dict]] = {}
    for relocation in contract.get("relocations", []):
        relocations_by_source.setdefault(relocation["source"], []).append(relocation)

    for relative, blocks in contract["preserved_blocks"].items():
        source_blocks: list[str] = []
        moved_indices = {
            index
            for relocation in relocations_by_source.get(relative, [])
            for index in relocation.get("block_indices", [])
        }
        split_relocations = {
            relocation["split_block_index"]: relocation
            for relocation in relocations_by_source.get(relative, [])
            if "split_block_index" in relocation
        }
        for index, block in enumerate(blocks):
            if index in moved_indices:
                continue
            if index in split_relocations:
                marker = split_relocations[index]["split_marker"]
                prefix, separator, _ = block.partition(marker)
                require(bool(separator), f"preservation split marker missing from contract: {relative}")
                source_blocks.append(prefix)
                continue
            source_blocks.append(block)
        require_blocks_in_order(relative, source_blocks, relative)

    for relocation in contract.get("relocations", []):
        source = relocation["source"]
        original_blocks = contract["preserved_blocks"][source]
        destination_blocks = [original_blocks[index] for index in relocation.get("block_indices", [])]
        if "split_block_index" in relocation:
            block = original_blocks[relocation["split_block_index"]]
            marker = relocation["split_marker"]
            _, separator, suffix = block.partition(marker)
            require(bool(separator), f"preservation split marker missing from contract: {source}")
            destination_blocks.insert(0, marker + suffix)
        require_blocks_in_order(relocation["destination"], destination_blocks, source)


def verify_pair_clarifications() -> None:
    summary = json.loads(SUMMARY.read_text())
    results = compact(read("sections/05_results.tex"))
    appendix = compact(read("appendix/K_pair_object.tex"))
    baseline = summary["baseline_background"]
    require(abs(baseline["fpr"] - baseline["accepted"] / baseline["denominator"]) < 1e-15,
            "baseline fake-rate arithmetic drift")
    require(
        f"{baseline['fpr'] * 100:.4f}" in results or f"{baseline['fpr'] * 100:.4f}" in appendix,
        "baseline achieved fake rate missing",
    )
    require("not at matched achieved rates" in appendix, "achieved-rate distinction missing")
    require("conditional two-sided" in appendix and "discordant decisions $b+c$" in appendix,
            "conditional uncertainty definition missing")
    require(f"split seed {summary['data_split_seed']}" in appendix, "common pair data split missing")
    require("first pair in deterministic pair order" in appendix, "regional representative-pair rule missing")
    for n in summary["regional_signal_denominators"].values():
        require(f"{n:,}".replace(",", "{,}") in appendix, "regional sample size missing")
    require(r"\mathcal{P}^{\mathrm{train}}_e" in appendix, "retained training-pair set missing")
    require(
        results.count("exceeded") + appendix.count("exceeded") == 1,
        "rate-limit qualification must appear exactly once",
    )

def verify_historical_findings() -> None:
    """Recompute the added historical findings from the public result files."""
    import statistics
    import hashlib

    def rows(relative: str) -> list[dict[str, str]]:
        with (ROOT / "results" / relative).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    appendix = compact(read("appendix/K_pair_object.tex"))
    expected: list[str] = []
    screen = rows("representation_reproduction/seed42_scalar_coarse_screen.csv")
    require(len(screen) == 22 and sum(r["screen_decision"] == "reproduced_across_three_seeds" for r in screen) == 3,
            "historical representation screen count drift")
    representation = {r["candidate"]: r for r in rows("representation_reproduction/global_summary.csv")}
    no_pt = float(representation["raw_coarse_cells_without_member_pt"]["mean_inclusive_signal_efficiency"])
    with_pt = float(representation["raw_coarse_cells_with_member_pt"]["mean_inclusive_signal_efficiency"])
    expected.extend(f"{v * 100:.3f}" for v in [no_pt, with_pt, with_pt - no_pt])
    expected.extend(f"{float(r['mean_validation_point_difference']) * 100:.2f}" for r in rows("representation_study/gain_source_effects.csv"))
    outputs = [r for r in rows("representation_study/representation_validation.csv") if r["study"] == "output_fusion"]
    require(len(outputs) == 3, "historical output/fusion comparison incomplete")
    expected.extend(f"{float(r['mean_inclusive_efficiency']) * 100:.3f}" for r in outputs)
    expected.extend(f"{(float(outputs[i]['mean_inclusive_efficiency']) - float(outputs[i-1]['mean_inclusive_efficiency'])) * 100:+.3f}" for i in [1, 2])
    factorial = [r for r in rows("pt_factorial/summary.csv") if r["candidate"] == "member_pt_and_event_pt"]
    require(len(factorial) == 2, "historical pT input comparison incomplete")
    expected.extend(f"{float(r['mean_delta_vs_baseline_percentage_points']):.2f}" for r in factorial)
    context = {r["candidate"]: r for r in rows("pt_context/summary.csv")}
    require(len(context) == 9, "historical seven-context comparison incomplete")
    expected.extend(f"{float(context[k]['mean_inclusive_signal_efficiency']) * 100:.3f}" for k in ["current_member_pt_and_event_second_pt", "pair_sum_and_balance"])
    expected.append(f"{float(context['pair_sum_and_balance']['delta_vs_current']) * 100:.3f}")
    history = rows("historical_validation/additional_results.csv")
    require(len(history) == 8, "historical additional-results population drift")
    for r in history:
        require(r["evidence_status"] == "historical_adaptive_validation", "historical evidence status drift")
        require(abs(float(r["signal_efficiency"]) - float(r["reference_efficiency"]) - float(r["difference"])) < 1e-14,
                "historical comparator arithmetic drift")
    for name in ["symmetric_fusion", "whole_pair_mlp", "em2_width", "all_finite_event_average"]:
        subset = [r for r in history if r["candidate"] == name]
        expected.append(f"{statistics.mean(float(r['signal_efficiency']) for r in subset) * 100:.3f}")
        if name in ["em2_width", "all_finite_event_average"]:
            expected.append(f"{statistics.mean(float(r['difference']) for r in subset) * 100:+.3f}")
        expected.append(f"{statistics.mean(float(r['reference_efficiency']) for r in subset) * 100:.3f}")
    provenance = json.loads((ROOT / "results/historical_validation/provenance.json").read_text())
    require(hashlib.sha256((ROOT / "results/historical_validation/additional_results.csv").read_bytes()).hexdigest() == provenance["additional_results_csv_sha256"],
            "historical aggregate evidence hash drift")
    for value in expected:
        require(value in appendix, f"historical report value missing or changed: {value}")
    require("historical adaptive-validation results" in appendix and "not independent confirmation" in appendix,
            "historical validation qualification missing")
    require("their separate contributions were not isolated" in appendix,
            "combined population/loss comparison must not imply isolated contributions")
    require("common physical grid" in appendix and "removes layer identity" in appendix,
            "cross-layer spatial compatibility qualification missing")


def main() -> int:
    verify_structure()
    verify_original_preservation()
    verify_pair_clarifications()
    verify_historical_findings()
    verify_citations()
    verify_evidence_paths()
    verify_numerical_contract()
    verify_terminology_contract()
    print("Report contract verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
