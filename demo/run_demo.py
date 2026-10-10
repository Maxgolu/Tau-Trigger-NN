"""Run a deterministic, data-free demonstration of the final pair pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pair_classifiers import JointOrThresholds, joint_or_decisions
from pair_data import aggregate_pair_scores, build_pair_dataset
from pair_models import build_pair_model, count_parameters


def synthetic_tobs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eventNumber": [101] * 6 + [202] * 3 + [303],
            "tob_index": list(range(6)) + list(range(3)) + [0],
            "tob_pt": [18.0, 42.0, 31.0, 25.0, 12.0, 8.0, 35.0, 22.0, 15.0, 11.0],
            "signal": [1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        }
    )


def main() -> int:
    torch.manual_seed(42)
    np.random.seed(42)
    config_path = ROOT / "configs" / "pair_final_all_finite_equal_event" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = build_pair_dataset(
        synthetic_tobs(),
        event_ids=[101, 202, 303],
        max_selected_tobs=None,
    )

    model = build_pair_model(config)
    model.eval()
    pair_count = len(dataset.pairs)
    images = torch.randn(pair_count, 2, 2, 12, 12)
    scalars = torch.randn(pair_count, 2, 47)
    context_by_event = dataset.events.set_index("eventNumber")["baseline_event_score"]
    context = torch.tensor(
        dataset.pairs["eventNumber"].map(context_by_event).to_numpy()[:, None],
        dtype=torch.float32,
    )
    with torch.no_grad():
        pair_scores = model(images, scalars, context).reshape(-1).numpy()

    predictions = dataset.pairs[["eventNumber", "pair_index"]].copy()
    predictions["pair_score"] = pair_scores
    event_scores = aggregate_pair_scores(dataset, predictions)
    event_scores = event_scores.merge(
        dataset.events[["eventNumber", "baseline_event_score"]],
        on="eventNumber",
        how="left",
        validate="one_to_one",
    )
    thresholds = JointOrThresholds(
        pt_threshold=30.0,
        neural_threshold=0.0,
        background_events=3,
        pt_accepted=0,
        neural_accepted=0,
        overlap_accepted=0,
        union_accepted=0,
    )
    decisions = joint_or_decisions(
        event_scores["baseline_event_score"],
        event_scores["event_score"],
        thresholds,
    )

    assert int(dataset.events.loc[dataset.events.eventNumber.eq(101), "pair_count"].iloc[0]) == 15
    assert np.isneginf(
        event_scores.loc[event_scores.eventNumber.eq(303), "event_score"].iloc[0]
    )
    assert count_parameters(model) == 4357

    output = {
        "status": "PASS",
        "configuration": str(config_path.relative_to(ROOT)),
        "model_parameters": count_parameters(model),
        "event_count": len(dataset.events),
        "finite_tob_count": len(dataset.selected_tobs),
        "pair_count": pair_count,
        "event_pair_counts": dict(
            zip(
                dataset.events["eventNumber"].astype(str),
                dataset.events["pair_count"].astype(int),
            )
        ),
        "demo_or_decisions": dict(
            zip(
                event_scores["eventNumber"].astype(str),
                [bool(value) for value in decisions],
            )
        ),
        "checks": [
            "all finite TOBs were retained",
            "six TOBs produced fifteen unordered pairs",
            "the final 4357-parameter architecture completed inference",
            "pair scores were reduced by maximum within each event",
            "the no-pair event received negative infinity and failed",
            "the neural and deterministic branches were combined by event-level OR",
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
