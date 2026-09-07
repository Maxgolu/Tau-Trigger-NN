"""Feature mappings for two canonically ordered TOBs.

The caller supplies the higher-pT member first and the lower-pT member second.
Canonical ordering itself belongs to :mod:`pair_data`; this module only maps
the already ordered calorimeter inputs into model-ready representations.
"""

import numpy as np

CORE_SHAPE = (5, 3, 3)
EM2_SHAPE = (12, 12)
SUMMARY_WIDTH = 16
COMPACT_EM2_FEATURE_NAMES = (
    "em2_max1",
    "em2_max2",
    "em2_max_neighbors_sum",
    "em2_width",
    "em2_normalized_width",
    "em2_best_3x3_fraction",
    "em2_maxratio_approx",
    "em2_top3_window1_fraction",
    "em2_top3_window2_fraction",
    "em2_top3_window3_fraction",
    "em2_top3_window12_sqdist",
    "em2_top3_window13_sqdist",
    "em2_top3_window23_sqdist",
    "em2_top2_3x3_sqdist",
    "em2_6x6_maxdist",
    "em2_outside_best_3x3_over_pt",
    "measured_tob_pt",
)


def _validated_array(values, trailing_shape, name):
    result = np.asarray(values, dtype=np.float32)
    if result.shape[-len(trailing_shape):] != trailing_shape:
        raise ValueError(
            f"{name} must end with shape {trailing_shape}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _matching_prefix(first, second, first_name, second_name):
    if first.shape[:-len(CORE_SHAPE)] != second.shape[:-len(CORE_SHAPE)]:
        raise ValueError(
            f"{first_name} and {second_name} must describe the same number of pairs"
        )


def _member_pt(values, prefix, name):
    result = np.asarray(values, dtype=np.float32)
    if result.shape == prefix + (1,):
        result = result[..., 0]
    if result.shape != prefix:
        raise ValueError(f"{name} must have shape {prefix} or {prefix + (1,)}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return result[..., None]


def em2_best_3x3_fraction(em2):
    """Return the energy fraction in the strongest contiguous 3x3 window.

    All 100 windows in a 12x12 image are considered. When the absolute total
    energy is at most ``1e-12``, the fraction is defined as zero.
    """
    image = _validated_array(em2, EM2_SHAPE, "em2")
    windows = np.lib.stride_tricks.sliding_window_view(
        image, (3, 3), axis=(-2, -1)
    )
    best = windows.sum(axis=(-2, -1)).max(axis=(-2, -1))
    total = image.sum(axis=(-2, -1))
    result = np.zeros_like(total, dtype=np.float32)
    np.divide(best, total, out=result, where=np.abs(total) > 1e-12)
    return result


def member_summary_features(core, em2):
    """Return ``[active counts, sums, dominance, EM2 fraction]``.

    The first fifteen values describe the five coarse calorimeter layers in
    EM0, EM1, EM2, EM3, HAD order. For each layer, the active-cell count uses
    the strict threshold ``cell > 0.1 * layer_max`` and dominance is
    ``2 * layer_max - layer_sum``.
    """
    cells = _validated_array(core, CORE_SHAPE, "core")
    image = _validated_array(em2, EM2_SHAPE, "em2")
    if cells.shape[:-len(CORE_SHAPE)] != image.shape[:-len(EM2_SHAPE)]:
        raise ValueError("core and em2 must describe the same TOBs")

    # Evaluate one layer at a time to preserve the registered single-object
    # reduction order exactly.
    prefix = cells.shape[:-len(CORE_SHAPE)]
    registered_layout = (
        cells.reshape(-1, 45).copy(order="F").reshape(-1, 5, 3, 3)
    )
    layer_sum = np.stack(
        [registered_layout[:, layer].sum(axis=(1, 2)) for layer in range(5)],
        axis=-1,
    )
    layer_max = np.stack(
        [registered_layout[:, layer].max(axis=(1, 2)) for layer in range(5)],
        axis=-1,
    )
    # Preserve the established single-object arithmetic order exactly:
    # maximum minus the sum of the other eight cells.
    dominance = layer_max - (layer_sum - layer_max)
    active = np.stack(
        [
            (
                registered_layout[:, layer]
                > 0.1 * layer_max[:, layer, None, None]
            ).sum(axis=(1, 2))
            for layer in range(5)
        ],
        axis=-1,
    )
    fraction = em2_best_3x3_fraction(image).reshape(-1, 1)
    result = np.concatenate(
        (
            active.astype(np.float32),
            layer_sum.astype(np.float32),
            dominance.astype(np.float32),
            fraction,
        ),
        axis=-1,
    )
    return result.reshape(prefix + (SUMMARY_WIDTH,))


def raw_cell_pair_features(left_core, right_core):
    """Return 45 left-member cells followed by 45 right-member cells."""
    left = _validated_array(left_core, CORE_SHAPE, "left_core")
    right = _validated_array(right_core, CORE_SHAPE, "right_core")
    _matching_prefix(left, right, "left_core", "right_core")
    prefix = left.shape[:-len(CORE_SHAPE)]
    return np.concatenate(
        (left.reshape(prefix + (45,)), right.reshape(prefix + (45,))), axis=-1
    )


def raw_cell_pt_pair_features(left_core, left_pt, right_core, right_pt):
    """Return ``[left cells, left pT, right cells, right pT]`` (92 values)."""
    cells = raw_cell_pair_features(left_core, right_core)
    prefix = cells.shape[:-1]
    left_values = cells[..., :45]
    right_values = cells[..., 45:]
    return np.concatenate(
        (
            left_values,
            _member_pt(left_pt, prefix, "left_pt"),
            right_values,
            _member_pt(right_pt, prefix, "right_pt"),
        ),
        axis=-1,
    )


def summary_pair_features(left_core, left_em2, right_core, right_em2):
    """Return the 16 left summaries followed by the 16 right summaries."""
    left = member_summary_features(left_core, left_em2)
    right = member_summary_features(right_core, right_em2)
    if left.shape[:-1] != right.shape[:-1]:
        raise ValueError("left and right inputs must describe the same number of pairs")
    return np.concatenate((left, right), axis=-1)


def summary_pt_pair_features(
    left_core,
    left_em2,
    left_pt,
    right_core,
    right_em2,
    right_pt,
):
    """Return ``[left summaries, left pT, right summaries, right pT]``."""
    summaries = summary_pair_features(
        left_core, left_em2, right_core, right_em2
    )
    prefix = summaries.shape[:-1]
    return np.concatenate(
        (
            summaries[..., :SUMMARY_WIDTH],
            _member_pt(left_pt, prefix, "left_pt"),
            summaries[..., SUMMARY_WIDTH:],
            _member_pt(right_pt, prefix, "right_pt"),
        ),
        axis=-1,
    )


def _safe_exact_zero_ratio(numerator, denominator):
    """Return a float32 ratio, defining only an exact-zero denominator as zero."""
    numerator64 = np.asarray(numerator, dtype=np.float64)
    denominator64 = np.asarray(denominator, dtype=np.float64)
    result = np.zeros(
        np.broadcast_shapes(numerator64.shape, denominator64.shape),
        dtype=np.float64,
    )
    np.divide(
        numerator64,
        denominator64,
        out=result,
        where=denominator64 != 0.0,
    )
    return result.astype(np.float32)


def _compact_em2_chunk(images, member_pt):
    """Vectorized implementation of the compact EM2 feature contract."""
    count = len(images)
    totals = images.sum(axis=(1, 2), dtype=np.float32)
    flat = images.reshape(count, 144)

    # Stable row-major ordering makes ties deterministic.
    cell_order = np.argsort(-flat, axis=1, kind="stable")[:, :2]
    cell_values = np.take_along_axis(flat, cell_order, axis=1)
    center_row, center_column = np.divmod(cell_order[:, 0], 12)

    grid_row, grid_column = np.ogrid[:12, :12]
    distance2 = (
        (grid_row[None] - center_row[:, None, None]) ** 2
        + (grid_column[None] - center_column[:, None, None]) ** 2
    )
    raw_width = np.sum(
        images * distance2,
        axis=(1, 2),
        dtype=np.float64,
    ).astype(np.float32)

    padded = np.pad(images, ((0, 0), (1, 1), (1, 1)))
    neighborhoods = np.lib.stride_tricks.sliding_window_view(
        padded, (3, 3), axis=(1, 2)
    )
    neighborhood_sums = neighborhoods.sum(axis=(-1, -2), dtype=np.float32)
    neighbor_sum = (
        neighborhood_sums[np.arange(count), center_row, center_column]
        - cell_values[:, 0]
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        images, (3, 3), axis=(1, 2)
    ).sum(axis=(-1, -2), dtype=np.float32)
    ranked = np.argsort(-windows.reshape(count, 100), axis=1, kind="stable")
    chosen_rows = np.full((count, 3), -1, dtype=np.int64)
    chosen_columns = np.full((count, 3), -1, dtype=np.int64)
    chosen_energy = np.empty((count, 3), dtype=np.float32)
    chosen_count = np.zeros(count, dtype=np.int8)
    row_index = np.arange(count)
    for rank in range(100):
        candidate = ranked[:, rank]
        row, column = np.divmod(candidate, 10)
        eligible = chosen_count < 3
        for slot in range(3):
            present = chosen_count > slot
            eligible &= (~present) | (
                (np.abs(row - chosen_rows[:, slot]) >= 3)
                | (np.abs(column - chosen_columns[:, slot]) >= 3)
            )
        for slot in range(3):
            mask = eligible & (chosen_count == slot)
            chosen_rows[mask, slot] = row[mask]
            chosen_columns[mask, slot] = column[mask]
            chosen_energy[mask, slot] = windows[
                row_index[mask], row[mask], column[mask]
            ]
        chosen_count[eligible] += 1
    if np.any(chosen_count != 3):
        raise ValueError("could not select three non-overlapping EM2 windows")

    top_fraction = _safe_exact_zero_ratio(chosen_energy, totals[:, None])
    top_distances = np.stack(
        (
            (chosen_rows[:, 0] - chosen_rows[:, 1]) ** 2
            + (chosen_columns[:, 0] - chosen_columns[:, 1]) ** 2,
            (chosen_rows[:, 0] - chosen_rows[:, 2]) ** 2
            + (chosen_columns[:, 0] - chosen_columns[:, 2]) ** 2,
            (chosen_rows[:, 1] - chosen_rows[:, 2]) ** 2
            + (chosen_columns[:, 1] - chosen_columns[:, 2]) ** 2,
        ),
        axis=1,
    ).astype(np.float32)

    reduced = images.reshape(count, 6, 2, 6, 2).sum(
        axis=(2, 4), dtype=np.float32
    ).reshape(count, 36)
    reduced_order = np.argsort(-reduced, axis=1, kind="stable")[:, :2]
    reduced_rows, reduced_columns = np.divmod(reduced_order, 6)
    six_by_six_distance = (
        (reduced_rows[:, 0] - reduced_rows[:, 1]) ** 2
        + (reduced_columns[:, 0] - reduced_columns[:, 1]) ** 2
    ).astype(np.float32)

    features = np.stack(
        (
            cell_values[:, 0],
            cell_values[:, 1],
            neighbor_sum,
            raw_width,
            _safe_exact_zero_ratio(raw_width, totals),
            _safe_exact_zero_ratio(chosen_energy[:, 0], totals),
            cell_values[:, 1] - cell_values[:, 0],
            top_fraction[:, 0],
            top_fraction[:, 1],
            top_fraction[:, 2],
            top_distances[:, 0],
            top_distances[:, 1],
            top_distances[:, 2],
            top_distances[:, 0],
            six_by_six_distance,
            _safe_exact_zero_ratio(
                totals - chosen_energy[:, 0], member_pt
            ),
            member_pt,
        ),
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("compact EM2 features contain a non-finite value")
    return features


def compact_em2_member_features(em2, member_pt, *, chunk_size=8192):
    """Return the exact 17-value compact EM2-plus-pT member representation.

    The historical ``em2_maxdist`` quantity is deliberately excluded. The
    retained values describe cell peaks, local energy, width, three
    non-overlapping 3x3 windows, coarse 6x6 separation, an outside-window/pT
    ratio and measured member pT. Inputs may have any shared prefix before the
    12x12 image axes; the result has that prefix followed by width 17.
    """
    image = _validated_array(em2, EM2_SHAPE, "em2")
    prefix = image.shape[:-len(EM2_SHAPE)]
    pt = np.asarray(member_pt, dtype=np.float32)
    if pt.shape == prefix + (1,):
        pt = pt[..., 0]
    if pt.shape != prefix:
        raise ValueError(f"member_pt must have shape {prefix} or {prefix + (1,)}")
    if not np.isfinite(pt).all():
        raise ValueError("member_pt contains a non-finite value")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    flat_images = image.reshape(-1, 12, 12)
    flat_pt = pt.reshape(-1)
    parts = [
        _compact_em2_chunk(flat_images[start : start + chunk_size], flat_pt[start : start + chunk_size])
        for start in range(0, len(flat_images), chunk_size)
    ]
    values = (
        np.concatenate(parts, axis=0)
        if parts
        else np.empty((0, len(COMPACT_EM2_FEATURE_NAMES)), dtype=np.float32)
    )
    return values.reshape(prefix + (len(COMPACT_EM2_FEATURE_NAMES),))
