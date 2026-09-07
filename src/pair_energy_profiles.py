"""Energy-region summaries for paired model-versus-baseline event decisions.

The conditional matched-member profile is an event-level diagnostic.  It uses
events containing an operational label-2 pair and bins each event by the lower
of the two TOB-associated ``truth_pt`` values for one deterministic pair.

This profile is *not* inclusive generator-level di-tau efficiency.  The export
does not identify distinct generator taus, so the two associated values may
refer to duplicate matches.  Truth-associated values are used only for
post-training analysis and are never model inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnergyRegion:
    """One half-open energy interval, expressed in GeV."""

    name: str
    low_gev: float
    high_gev: float

    def contains(self, values: np.ndarray) -> np.ndarray:
        return (values >= self.low_gev) & (values < self.high_gev)


@dataclass(frozen=True)
class ConditionalMatchedMemberProfile:
    """Regional results plus an accounting table for the conditional sample."""

    regions: pd.DataFrame
    population: pd.DataFrame

    @property
    def terminology(self) -> str:
        """Return the display name for this conditional diagnostic."""

        return CONDITIONAL_MATCHED_MEMBER_TERMINOLOGY

    @property
    def limitations(self) -> tuple[str, ...]:
        """Return the scientific limitations that must accompany the result."""

        return CONDITIONAL_MATCHED_MEMBER_LIMITATIONS


CONDITIONAL_MATCHED_MEMBER_TERMINOLOGY = (
    "conditional event efficiency versus the lower TOB-associated member truth_pt"
)

CONDITIONAL_MATCHED_MEMBER_LIMITATIONS = (
    (
        "The population contains events with an operational label-2 pair; it is "
        "not an inclusive signal-sample efficiency denominator."
    ),
    (
        "Positive TOB labels do not verify that the members match two distinct "
        "generator-level tau particles."
    ),
    (
        "Duplicate truth matches may occur; associated truth_pt is used only to "
        "bin events for analysis and is never an inference input."
    ),
)

DEFAULT_REGIONS = (
    EnergyRegion("low_10_25", 10.0, 25.0),
    EnergyRegion("medium_25_60", 25.0, 60.0),
    EnergyRegion("high_60_plus", 60.0, float("inf")),
)

CONDITIONAL_MATCHED_MEMBER_EVENT_COLUMNS = (
    "global_event_id",
    "model_seed",
    "member_l_associated_truth_pt_gev",
    "member_s_associated_truth_pt_gev",
    "model_pass",
    "baseline_pass",
    "has_operational_label2_pair",
)


def _boolean_array(values: Iterable[object], name: str) -> np.ndarray:
    series = pd.Series(values, copy=False)
    if series.isna().any() or not series.isin((True, False)).all():
        raise ValueError(f"{name} must contain only Boolean values")
    return series.to_numpy(dtype=bool)


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise KeyError(f"missing {name} columns: {sorted(missing)}")


def _require_validation_rows(frame: pd.DataFrame, name: str) -> None:
    """Reject mixed splits when a caller supplies explicit split metadata."""

    if "split" in frame.columns and not frame["split"].eq("validation").all():
        raise ValueError(f"{name} must contain validation rows only")


def build_conditional_matched_member_events(
    validation_pairs: pd.DataFrame,
    validation_objects: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("model_seed",),
    event_column: str = "global_event_id",
    pair_column: str = "pair_index",
    member_l_object_column: str = "tob_index_a",
    member_s_object_column: str = "tob_index_b",
    member_l_rank_column: str = "selected_rank_a",
    member_s_rank_column: str = "selected_rank_b",
    object_index_column: str = "tob_index",
    object_label_column: str = "signal",
    object_truth_pt_column: str = "truth_pt",
    truth_pt_unit: str = "MeV",
    model_pass_column: str = "model_pass",
    baseline_pass_column: str = "baseline_pass",
) -> pd.DataFrame:
    """Build the event table consumed by the conditional energy profile.

    ``validation_pairs`` must contain the frozen canonical pair identities and
    ``validation_objects`` must contain the corresponding TOB rows.  Operational
    pair labels are reconstructed from the two binary TOB labels; they are not
    inferred from ``truth_pt``.  If an event has several operational label-2
    pairs, the lowest ``pair_index`` is selected deterministically, matching the
    validation analysis used for the regional plots.

    The associated ``truth_pt`` values are post-training analysis coordinates.
    They do not establish that the two positive TOB rows correspond to distinct
    generator-level tau particles.
    """

    if not group_columns:
        raise ValueError("at least one decision grouping column is required")
    pair_columns = {
        event_column,
        pair_column,
        member_l_object_column,
        member_s_object_column,
        member_l_rank_column,
        member_s_rank_column,
    }
    object_columns = {
        event_column,
        object_index_column,
        object_label_column,
        object_truth_pt_column,
    }
    decision_columns = {
        event_column,
        model_pass_column,
        baseline_pass_column,
        *group_columns,
    }
    _require_columns(validation_pairs, pair_columns, "validation-pair")
    _require_columns(validation_objects, object_columns, "validation-object")
    _require_columns(decisions, decision_columns, "decision")
    _require_validation_rows(validation_pairs, "validation_pairs")
    _require_validation_rows(validation_objects, "validation_objects")
    _require_validation_rows(decisions, "decisions")

    pair_keys = [event_column, pair_column]
    if validation_pairs.duplicated(pair_keys).any():
        raise ValueError("validation pair identities must be unique")
    if validation_objects.duplicated([event_column, object_index_column]).any():
        raise ValueError("validation object identities must be unique")
    decision_keys = [*group_columns, event_column]
    if decisions.duplicated(decision_keys).any():
        raise ValueError("each event must have one decision per model group")
    if decisions.groupby(event_column, dropna=False)[baseline_pass_column].nunique(
        dropna=False
    ).gt(1).any():
        raise ValueError("baseline decisions must be identical across model groups")

    ranks_l = pd.to_numeric(
        validation_pairs[member_l_rank_column], errors="coerce"
    ).to_numpy(dtype=np.float64)
    ranks_s = pd.to_numeric(
        validation_pairs[member_s_rank_column], errors="coerce"
    ).to_numpy(dtype=np.float64)
    pair_indices = pd.to_numeric(
        validation_pairs[pair_column], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(ranks_l).all()
        or not np.isfinite(ranks_s).all()
        or not np.equal(ranks_l, np.floor(ranks_l)).all()
        or not np.equal(ranks_s, np.floor(ranks_s)).all()
        or (ranks_l < 0).any()
        or not (ranks_l < ranks_s).all()
    ):
        raise ValueError("pair members must use canonical L/S selected-rank order")
    if (
        not np.isfinite(pair_indices).all()
        or not np.equal(pair_indices, np.floor(pair_indices)).all()
        or (pair_indices < 0).any()
    ):
        raise ValueError("pair_index must contain finite non-negative integers")
    if (
        validation_pairs[member_l_object_column].to_numpy()
        == validation_pairs[member_s_object_column].to_numpy()
    ).any():
        raise ValueError("a pair must contain two different TOB identities")

    labels = pd.to_numeric(
        validation_objects[object_label_column], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("TOB operational labels must contain only 0 or 1")

    object_values = validation_objects.loc[:, list(object_columns)].copy()
    object_values[object_label_column] = labels.astype(np.uint8)
    left = object_values.rename(
        columns={
            object_index_column: member_l_object_column,
            object_label_column: "_member_l_label",
            object_truth_pt_column: "_member_l_truth_pt",
        }
    )
    right = object_values.rename(
        columns={
            object_index_column: member_s_object_column,
            object_label_column: "_member_s_label",
            object_truth_pt_column: "_member_s_truth_pt",
        }
    )
    pair_members = validation_pairs.loc[:, list(pair_columns)].merge(
        left,
        on=[event_column, member_l_object_column],
        how="left",
        validate="many_to_one",
        indicator="_left_join",
    )
    pair_members[pair_column] = pair_indices.astype(np.int64)
    if not pair_members["_left_join"].eq("both").all():
        raise ValueError("a canonical L member does not resolve to a validation object")
    pair_members = pair_members.drop(columns="_left_join").merge(
        right,
        on=[event_column, member_s_object_column],
        how="left",
        validate="many_to_one",
        indicator="_right_join",
    )
    if not pair_members["_right_join"].eq("both").all():
        raise ValueError("a canonical S member does not resolve to a validation object")
    pair_members = pair_members.drop(columns="_right_join")
    pair_members["_operational_pair_label"] = (
        pair_members["_member_l_label"].astype(np.uint8)
        + pair_members["_member_s_label"].astype(np.uint8)
    )
    if "three_class_label" in validation_pairs.columns:
        stored = pd.to_numeric(
            validation_pairs["three_class_label"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        rebuilt = pair_members["_operational_pair_label"].to_numpy(dtype=np.float64)
        if not np.array_equal(stored, rebuilt):
            raise ValueError("stored pair labels disagree with the two TOB labels")

    selected = (
        pair_members.loc[pair_members["_operational_pair_label"].eq(2)]
        .sort_values([event_column, pair_column], kind="mergesort")
        .drop_duplicates(event_column, keep="first")
        .copy()
    )
    if selected.empty:
        raise ValueError("no validation event contains an operational label-2 pair")

    member_l_truth = pd.to_numeric(
        selected["_member_l_truth_pt"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    member_s_truth = pd.to_numeric(
        selected["_member_s_truth_pt"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(member_l_truth).all() or not np.isfinite(member_s_truth).all():
        raise ValueError(
            "the selected operational label-2 pair requires two finite "
            "associated truth_pt values"
        )
    if truth_pt_unit == "MeV":
        scale = 1000.0
    elif truth_pt_unit == "GeV":
        scale = 1.0
    else:
        raise ValueError("truth_pt_unit must be 'MeV' or 'GeV'")
    selected["member_l_associated_truth_pt_gev"] = member_l_truth / scale
    selected["member_s_associated_truth_pt_gev"] = member_s_truth / scale

    model_pass = _boolean_array(decisions[model_pass_column], model_pass_column)
    baseline_pass = _boolean_array(
        decisions[baseline_pass_column], baseline_pass_column
    )
    decision_values = decisions.loc[:, list(decision_columns)].copy()
    decision_values[model_pass_column] = model_pass
    decision_values[baseline_pass_column] = baseline_pass
    selected_columns = [
        event_column,
        "member_l_associated_truth_pt_gev",
        "member_s_associated_truth_pt_gev",
    ]
    result = decision_values.merge(
        selected.loc[:, selected_columns],
        on=event_column,
        how="inner",
        validate="many_to_one",
    )
    selected_events = set(selected[event_column])
    expected_rows = 0
    for _, group in decision_values.groupby(
        group_columns[0] if len(group_columns) == 1 else list(group_columns),
        sort=False,
        dropna=False,
    ):
        if not selected_events.issubset(set(group[event_column])):
            raise ValueError("a model group is missing conditional event decisions")
        expected_rows += len(selected_events)
    if len(result) != expected_rows:
        raise AssertionError("conditional event reconstruction did not reconcile")

    result = result.rename(
        columns={
            model_pass_column: "model_pass",
            baseline_pass_column: "baseline_pass",
        }
    )
    result["has_operational_label2_pair"] = True
    output_columns = [
        event_column,
        *group_columns,
        "member_l_associated_truth_pt_gev",
        "member_s_associated_truth_pt_gev",
        "model_pass",
        "baseline_pass",
        "has_operational_label2_pair",
    ]
    result = result.loc[:, output_columns].sort_values(
        [*group_columns, event_column], kind="mergesort"
    )
    return result.reset_index(drop=True)


def paired_count_summary(
    model_pass: Iterable[object],
    baseline_pass: Iterable[object],
    *,
    confidence_z: float = 1.96,
) -> dict[str, int | float | str | None]:
    """Summarize paired decisions and their normal-approximation uncertainty.

    The unit is one event.  For every event, the paired difference is ``+1``
    when only the model accepts it, ``-1`` when only the baseline accepts it,
    and zero otherwise.  The standard error is the sample standard deviation
    of those paired differences divided by ``sqrt(n)``.
    """

    model = _boolean_array(model_pass, "model_pass")
    baseline = _boolean_array(baseline_pass, "baseline_pass")
    if model.shape != baseline.shape:
        raise ValueError("model and baseline decisions must be aligned")
    if not np.isfinite(confidence_z) or confidence_z <= 0:
        raise ValueError("confidence_z must be finite and positive")

    count = int(model.size)
    model_accepted = int(model.sum())
    baseline_accepted = int(baseline.sum())
    both = int((model & baseline).sum())
    model_only = int((model & ~baseline).sum())
    baseline_only = int((~model & baseline).sum())
    neither = count - both - model_only - baseline_only

    if count == 0:
        return {
            "denominator": 0,
            "model_accepted": 0,
            "baseline_accepted": 0,
            "both_accepted": 0,
            "model_only_accepted": 0,
            "baseline_only_accepted": 0,
            "neither_accepted": 0,
            "model_efficiency": None,
            "baseline_efficiency": None,
            "paired_difference": None,
            "paired_standard_error": None,
            "paired_ci_low": None,
            "paired_ci_high": None,
            "point_status": "empty",
            "evidence_status": "empty",
        }

    difference = model.astype(np.float64) - baseline.astype(np.float64)
    mean = float(difference.mean())
    standard_error = (
        float(difference.std(ddof=1) / sqrt(count)) if count > 1 else 0.0
    )
    ci_low = mean - confidence_z * standard_error
    ci_high = mean + confidence_z * standard_error
    point_status = "beats" if mean > 0 else "loses" if mean < 0 else "ties"
    evidence_status = (
        "beats" if ci_low > 0 else "loses" if ci_high < 0 else "uncertain"
    )
    return {
        "denominator": count,
        "model_accepted": model_accepted,
        "baseline_accepted": baseline_accepted,
        "both_accepted": both,
        "model_only_accepted": model_only,
        "baseline_only_accepted": baseline_only,
        "neither_accepted": neither,
        "model_efficiency": model_accepted / count,
        "baseline_efficiency": baseline_accepted / count,
        "paired_difference": mean,
        "paired_standard_error": standard_error,
        "paired_ci_low": ci_low,
        "paired_ci_high": ci_high,
        "point_status": point_status,
        "evidence_status": evidence_status,
    }


def summarize_conditional_matched_members(
    events: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("model_seed",),
    event_column: str = "global_event_id",
    member_l_truth_pt_column: str = "member_l_associated_truth_pt_gev",
    member_s_truth_pt_column: str = "member_s_associated_truth_pt_gev",
    model_pass_column: str = "model_pass",
    baseline_pass_column: str = "baseline_pass",
    operational_pair_column: str = "has_operational_label2_pair",
    regions: Sequence[EnergyRegion] = DEFAULT_REGIONS,
    confidence_z: float = 1.96,
) -> ConditionalMatchedMemberProfile:
    """Build the conditional matched-member regional profile.

    Each input row must represent one event for one model run.  The two member
    columns are TOB-associated truth values for the already selected,
    deterministic operational label-2 pair.  Their minimum is the analysis
    coordinate; the canonical L/S ordering itself is not reinterpreted as
    generator-level ordering.
    """

    required = {
        event_column,
        member_l_truth_pt_column,
        member_s_truth_pt_column,
        model_pass_column,
        baseline_pass_column,
        operational_pair_column,
        *group_columns,
    }
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"missing conditional-profile columns: {sorted(missing)}")
    if not group_columns:
        raise ValueError("at least one grouping column is required")
    if not regions:
        raise ValueError("at least one energy region is required")
    for previous, current in pairwise(regions):
        if previous.high_gev > current.low_gev:
            raise ValueError("energy regions must not overlap")
    for region in regions:
        if (
            not np.isfinite(region.low_gev)
            or np.isnan(region.high_gev)
            or region.high_gev <= region.low_gev
        ):
            raise ValueError("energy regions must have valid increasing bounds")

    work = events.copy()
    operational = _boolean_array(work[operational_pair_column], operational_pair_column)
    if not operational.all():
        raise ValueError(
            "conditional matched-member rows must all contain an operational "
            "label-2 pair"
        )
    member_l = work[member_l_truth_pt_column].to_numpy(dtype=np.float64)
    member_s = work[member_s_truth_pt_column].to_numpy(dtype=np.float64)
    if not np.isfinite(member_l).all() or not np.isfinite(member_s).all():
        raise ValueError("both TOB-associated truth_pt values must be finite")
    work["matched_member_min_truth_pt_gev"] = np.minimum(member_l, member_s)

    key_columns = [*group_columns, event_column]
    if work.duplicated(key_columns).any():
        raise ValueError("each event must appear exactly once within each model group")

    regional_rows: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []
    grouper: str | list[str]
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    grouped = work.groupby(grouper, sort=True, dropna=False)
    for raw_key, group in grouped:
        keys = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        identity = dict(zip(group_columns, keys))
        energy = group["matched_member_min_truth_pt_gev"].to_numpy(dtype=np.float64)
        assigned = np.zeros(len(group), dtype=bool)
        for region in regions:
            mask = region.contains(energy)
            assigned |= mask
            summary = paired_count_summary(
                group.loc[mask, model_pass_column],
                group.loc[mask, baseline_pass_column],
                confidence_z=confidence_z,
            )
            regional_rows.append(
                {
                    **identity,
                    "region": region.name,
                    "low_gev": region.low_gev,
                    "high_gev": region.high_gev,
                    **summary,
                }
            )
        population_rows.append(
            {
                **identity,
                "conditional_event_count": len(group),
                "reported_region_event_count": int(assigned.sum()),
                "outside_reported_regions": int((~assigned).sum()),
                "minimum_reported_energy_gev": float(
                    min(region.low_gev for region in regions)
                ),
            }
        )

    return ConditionalMatchedMemberProfile(
        regions=pd.DataFrame(regional_rows),
        population=pd.DataFrame(population_rows),
    )
