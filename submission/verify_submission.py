"""Verify release bindings, results, unified report, and data-free demo."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "submission" / "final_system_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in manifest["file_bindings"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"release binding drift: {relative}")

    summary = json.loads(
        (ROOT / "results" / "final_test" / "final_result_summary.json").read_text(
            encoding="utf-8"
        )
    )
    with (ROOT / "results" / "final_test" / "seed_results.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    efficiencies = [float(row["candidate_efficiency"]) for row in rows]
    baseline = [float(row["baseline_efficiency"]) for row in rows]
    if not math.isclose(
        statistics.mean(efficiencies),
        summary["inclusive_efficiency"]["candidate_seed_mean"],
        abs_tol=1e-15,
    ):
        raise RuntimeError("candidate mean drift")
    if not math.isclose(
        statistics.stdev(efficiencies),
        summary["inclusive_efficiency"]["candidate_seed_sample_sd"],
        abs_tol=1e-15,
    ):
        raise RuntimeError("candidate sample SD drift")
    if len(set(baseline)) != 1 or not math.isclose(
        baseline[0],
        summary["inclusive_efficiency"]["deterministic_baseline"],
        abs_tol=1e-15,
    ):
        raise RuntimeError("baseline drift")

    subprocess.run(
        [sys.executable, str(ROOT / "submission" / "verify_report.py")],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "demo" / "verify_demo.py")],
        cwd=ROOT,
        check=True,
    )
    print("Submission verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
