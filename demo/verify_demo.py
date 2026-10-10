"""Execute the public demo and compare it with the committed contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "demo" / "run_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(completed.stdout)
    expected = json.loads(
        (ROOT / "demo" / "expected_output.json").read_text(encoding="utf-8")
    )
    if actual != expected:
        print("Demo output differs from demo/expected_output.json", file=sys.stderr)
        print(completed.stdout, file=sys.stderr)
        return 1
    print("Demo verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
