from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.distributions.audit import build_split_audit
from src.distributions.data import (
    add_split_column,
    iter_aligned_batches,
    limit_events_per_sample,
    load_metadata,
    select_scope,
)
from src.distributions.plotting import plot_distribution, plot_pt_conditioned
from src.distributions.variables import (
    ALL_VARIABLES,
    EVENT_VARIABLES,
    OBJECT_VARIABLES,
    build_event_table,
    extract_object_variables,
    required_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "distributions" / "core_v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "distributions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tau-trigger data distributions")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Distribution JSON config")
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Run only the named plot IDs; omit to generate every configured plot",
    )
    parser.add_argument(
        "--scope", choices=["train", "validation", "val", "test", "train+validation", "all"],
        default=None, help="Override the configured data scope",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the configured split seed")
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT), help="Directory containing Signal/ and Background/")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Base output directory")
    parser.add_argument(
        "--max_events_per_sample", type=int, default=None,
        help="Optional small per-sample event limit for a CPU smoke test",
    )
    parser.add_argument("--skip_audit", action="store_true", help="Skip the configured split audit")
    parser.add_argument(
        "--audit_only", action="store_true",
        help="Write the split audit and exit without calculating plot variables",
    )
    return parser.parse_args()


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not config.get("run_id"):
        raise ValueError("Distribution config must define a short run_id")
    return config


def _all_tasks(config: dict) -> list[dict]:
    tasks: list[dict] = []
    for level, key in (("object", "object_plots"), ("event", "event_plots")):
        for task in config.get(key, []):
            task = dict(task)
            task["level"] = level
            if "id" not in task or "variable" not in task or "group_by" not in task:
                raise ValueError(f"Every {key} entry needs id, variable, and group_by: {task}")
            if task["variable"] not in ALL_VARIABLES:
                raise KeyError(f"Unknown distribution variable: {task['variable']}")
            if ALL_VARIABLES[task["variable"]].level != level:
                raise ValueError(f"Variable {task['variable']} is not a {level}-level variable")
            tasks.append(task)
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Plot IDs must be unique")
    return tasks


def _select_tasks(tasks: list[dict], requested_ids: list[str] | None) -> list[dict]:
    if requested_ids is None:
        return tasks
    available = {task["id"] for task in tasks}
    unknown = set(requested_ids).difference(available)
    if unknown:
        raise KeyError(f"Unknown plot IDs {sorted(unknown)}. Available IDs: {sorted(available)}")
    requested = set(requested_ids)
    return [task for task in tasks if task["id"] in requested]


def _calculate_object_table(
    metadata: pd.DataFrame,
    data_dir: str | Path,
    variable_names: set[str],
    batch_size: int,
) -> pd.DataFrame:
    sources = required_sources(variable_names)
    batches: list[pd.DataFrame] = []
    for aligned_batch in iter_aligned_batches(
        metadata,
        data_dir=data_dir,
        batch_size=batch_size,
        need_tensors="tensors" in sources,
        need_em2="em2" in sources,
    ):
        batches.append(extract_object_variables(aligned_batch, variable_names))
    if not batches:
        raise ValueError("No objects are available after applying the configured data scope")
    return pd.concat(batches, ignore_index=True)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    tasks = _select_tasks(_all_tasks(config), args.only)
    if not tasks:
        raise ValueError("No distribution plots were selected")

    scope = args.scope or config.get("data_scope", "train")
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    normalization = config.get("normalization", "probability")
    show_plots = bool(config.get("show_plots", False))
    batch_size = int(config.get("object_batch_size", 100_000))
    output_root = Path(args.output_dir).resolve() / config["run_id"]
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading object metadata...")
    full_metadata = load_metadata(args.data_dir)
    full_metadata = add_split_column(full_metadata, seed)

    audit_config = config.get("split_audit", {})
    if audit_config.get("enabled", False) and not args.skip_audit:
        seeds = [int(value) for value in audit_config.get("seeds", [42, 123, 456])]
        print(f"Auditing legacy splits for seeds {seeds}...")
        audit = build_split_audit(full_metadata, seeds)
        audit.to_csv(output_root / "split_audit.csv", index=False)
    elif args.audit_only:
        raise ValueError("--audit_only requires split_audit.enabled=true and no --skip_audit")

    if args.audit_only:
        print(f"Split audit saved to: {output_root / 'split_audit.csv'}")
        return

    metadata = select_scope(full_metadata, scope)
    metadata = limit_events_per_sample(metadata, args.max_events_per_sample, seed)
    print(
        f"Selected scope '{scope}': {metadata['event_uid'].nunique():,} events, "
        f"{len(metadata):,} objects"
    )

    object_variable_names = {
        task["variable"] for task in tasks if task["level"] == "object"
    }
    objects = _calculate_object_table(
        metadata, args.data_dir, object_variable_names, batch_size
    )
    events = build_event_table(objects) if any(task["level"] == "event" for task in tasks) else None

    run_summary = {
        "run_id": config["run_id"],
        "scope": scope,
        "seed": seed,
        "event_count": int(metadata["event_uid"].nunique()),
        "object_count": int(len(metadata)),
        "plots": {},
    }
    formats = config.get("output_formats", ["png"])
    if formats != ["png"]:
        raise ValueError("The current implementation supports output_formats: ['png']")

    pt_config = config.get("pt_conditioning", {})
    for task in tasks:
        plot_id = task["id"]
        variable_name = task["variable"]
        spec = ALL_VARIABLES[variable_name]
        frame = objects if task["level"] == "object" else events
        if frame is None:
            raise RuntimeError(f"Event table was not built for plot {plot_id}")
        options = task.get("options", {})
        print(f"Generating {plot_id}...")
        summary = plot_distribution(
            frame,
            spec,
            task["group_by"],
            output_root / f"{plot_id}.png",
            normalization,
            show_plots,
            options,
        )

        if (
            task["level"] == "object"
            and task.get("pt_conditioned", False)
            and pt_config.get("enabled", False)
        ):
            summary["pt_conditioned"] = plot_pt_conditioned(
                objects,
                OBJECT_VARIABLES[variable_name],
                [float(value) for value in pt_config["bins_gev"]],
                output_root / f"{plot_id}_pt_conditioned.png",
                normalization,
                show_plots,
                options,
            )
        run_summary["plots"][plot_id] = summary
        _write_json(output_root / f"{plot_id}.json", summary)

    _write_json(output_root / "run_summary.json", run_summary)
    with open(output_root / "config_used.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Finished. Outputs saved under: {output_root}")


if __name__ == "__main__":
    main()
