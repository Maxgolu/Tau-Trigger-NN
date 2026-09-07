"""Feature mappings for two canonically ordered TOBs.

The caller supplies the higher-pT member first and the lower-pT member second.
Canonical ordering itself belongs to :mod:`pair_data`; this module only maps
the already ordered calorimeter inputs into model-ready representations.
"""

import numpy as np

CORE_SHAPE = (5, 3, 3)
EM2_SHAPE = (12, 12)
SUMMARY_WIDTH = 16


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
