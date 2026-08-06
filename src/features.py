import numpy as np
import pandas as pd


def get_core_physics(df: pd.DataFrame) -> np.ndarray:
    """Extracts the 4 kinematic physics parameters from the flattened npz array."""
    cols = [f"feat_{i}" for i in range(4)]
    return df[cols].values


def get_core_tensors(df: pd.DataFrame) -> np.ndarray:
    """Extracts the 45 flat structural cell tensors."""
    cols = [f"tensor_{i}" for i in range(45)]
    return df[cols].values


def _get_em2_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Returns the EM2 cells reshaped as (N, 12, 12).
    Each row in the DataFrame becomes one 12x12 EM2 matrix.
    """
    cols = [f"em2_cell_{i}" for i in range(144)]

    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing EM2 columns, for example: {missing[:5]}")

    return df[cols].values.reshape(-1, 12, 12)


def _get_tob_pt(df: pd.DataFrame) -> np.ndarray:
    """
    Returns TOB pT as shape (N, 1).

    Prefer tob_pt if it exists.
    If not, use feat_1, because in the original code X_feats[:, :, 1] is used as pt.
    """
    if "tob_pt" in df.columns:
        pt = df["tob_pt"].values
    elif "feat_1" in df.columns:
        pt = df["feat_1"].values
    else:
        raise KeyError("Could not find tob_pt or feat_1 for pT normalization.")

    pt = pt.reshape(-1, 1)

    # Avoid division by zero
    pt = np.where(pt == 0, 1e-8, pt)

    return pt


def _all_3x3_window_sums(em2_3d: np.ndarray) -> np.ndarray:
    """
    Computes sums of all possible 3x3 windows inside each 12x12 EM2 matrix.

    Input:
        em2_3d shape: (N, 12, 12)

    Output:
        window_sums shape: (N, 10, 10)

    Explanation:
        A 3x3 window can start at row 0..9 and col 0..9.
    """
    N = em2_3d.shape[0]
    window_sums = np.zeros((N, 10, 10), dtype=np.float32)

    for y in range(10):
        for x in range(10):
            window_sums[:, y, x] = em2_3d[:, y:y + 3, x:x + 3].sum(axis=(1, 2))

    return window_sums


def get_em2_max(df: pd.DataFrame) -> np.ndarray:
    """Dynamically finds the maximum pixel in the EM2 tensor."""
    em2_3d = _get_em2_matrix(df)
    max_vals = em2_3d.reshape(em2_3d.shape[0], -1).max(axis=1)
    return max_vals.reshape(-1, 1)


def get_em2_max_neighbors_sum(df: pd.DataFrame) -> np.ndarray:
    """
    Finds the maximum pixel in the 12x12 EM2 tensor and calculates the sum
    of its 8 surrounding neighbors.

    This does not include the max pixel itself.
    """
    em2_3d = _get_em2_matrix(df)
    N_events = em2_3d.shape[0]

    flat_max_indices = em2_3d.reshape(N_events, -1).argmax(axis=1)
    y_coords, x_coords = np.unravel_index(flat_max_indices, (12, 12))

    padded_em2 = np.pad(
        em2_3d,
        pad_width=((0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=0
    )

    y_pad = y_coords + 1
    x_pad = x_coords + 1

    neighbor_sums = np.zeros((N_events, 1), dtype=np.float32)

    for i in range(N_events):
        y, x = y_pad[i], x_pad[i]

        window_sum = np.sum(padded_em2[i, y - 1:y + 2, x - 1:x + 2])
        center_val = padded_em2[i, y, x]

        neighbor_sums[i, 0] = window_sum - center_val

    return neighbor_sums


def get_all_em2_144(df: pd.DataFrame) -> np.ndarray:
    """
    Returns every high-res EM2 pixel, all 144 of them.
    """
    cols = [f"em2_cell_{i}" for i in range(144)]
    return df[cols].values


def get_em2_best_3x3_fraction(df: pd.DataFrame) -> np.ndarray:
    """
    Feature:
        strongest 3x3 EM2 window / total EM2 energy

    Physics meaning:
        Measures how concentrated the EM2 shower is in the strongest local region.
        A high value means most of the EM2 energy is localized in one compact area.
    """
    em2_3d = _get_em2_matrix(df)

    total_em2 = em2_3d.sum(axis=(1, 2)).reshape(-1, 1)
    total_em2 = np.where(total_em2 == 0, 1e-8, total_em2)

    window_sums = _all_3x3_window_sums(em2_3d)
    best_3x3 = window_sums.max(axis=(1, 2)).reshape(-1, 1)

    return best_3x3 / total_em2


def get_em2_outside_best_3x3_over_pt(df: pd.DataFrame) -> np.ndarray:
    """
    Feature:
        energy outside strongest 3x3 EM2 window / tob_pt

    Physics meaning:
        Measures how much EM2 energy is outside the leading compact core,
        normalized by the candidate pT.

        For a tau-like compact shower, this value is expected to be smaller.
        For a more diffuse jet-like background, this value may be larger.
    """
    em2_3d = _get_em2_matrix(df)

    total_em2 = em2_3d.sum(axis=(1, 2)).reshape(-1, 1)

    window_sums = _all_3x3_window_sums(em2_3d)
    best_3x3 = window_sums.max(axis=(1, 2)).reshape(-1, 1)

    outside_energy = total_em2 - best_3x3

    pt = _get_tob_pt(df)

    return outside_energy / pt


def get_em2_top3_3x3_features(df: pd.DataFrame) -> np.ndarray:
    """
    Features based on the top 3 non-overlapping 3x3 EM2 windows.

    Returns 6 features:
        1. best_3x3_sum / total_EM2
        2. second_3x3_sum / total_EM2
        3. third_3x3_sum / total_EM2
        4. distance between best and second 3x3 centers
        5. distance between best and third 3x3 centers
        6. distance between second and third 3x3 centers

    Physics meaning:
        Measures whether there are several strong local regions,
        and whether these regions are close together or spread out.
    """
    em2_3d = _get_em2_matrix(df)
    N = em2_3d.shape[0]

    total_em2 = em2_3d.sum(axis=(1, 2))
    total_em2 = np.where(total_em2 == 0, 1e-8, total_em2)

    window_sums = _all_3x3_window_sums(em2_3d)

    top_energies = np.zeros((N, 3), dtype=np.float32)
    top_centers = np.zeros((N, 3, 2), dtype=np.float32)

    for i in range(N):
        ws = window_sums[i].copy()

        for k in range(3):
            flat_idx = np.argmax(ws)
            y, x = np.unravel_index(flat_idx, ws.shape)

            top_energies[i, k] = ws[y, x]

            # Center of the 3x3 window.
            # If the window starts at (y, x), its center is at (y+1, x+1).
            top_centers[i, k] = [y + 1, x + 1]

            # Remove overlapping windows.
            # A 3x3 window overlaps with another if its start position is too close.
            y_min = max(0, y - 2)
            y_max = min(10, y + 3)
            x_min = max(0, x - 2)
            x_max = min(10, x + 3)

            ws[y_min:y_max, x_min:x_max] = -np.inf

    energy_fractions = top_energies / total_em2.reshape(-1, 1)

    d12 = np.linalg.norm(top_centers[:, 0, :] - top_centers[:, 1, :], axis=1)
    d13 = np.linalg.norm(top_centers[:, 0, :] - top_centers[:, 2, :], axis=1)
    d23 = np.linalg.norm(top_centers[:, 1, :] - top_centers[:, 2, :], axis=1)

    distances = np.stack([d12, d13, d23], axis=1)

    return np.hstack([energy_fractions, distances])


def get_5_layers_3x3(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts the 45 flat tensor columns and reshapes them into
    5 layers of 3x3 matrices (EM0, EM1, EM2, EM3, HAD).

    Output shape: (N_events, 5, 3, 3)
    Index mapping:
        0: EM0
        1: EM1
        2: EM2
        3: EM3
        4: HAD
    """
    cols = [f"tensor_{i}" for i in range(45)]

    # Check if columns exist to prevent silent failures
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing basic tensor columns. Example: {missing[:3]}")

    # Reshape: N events, 5 layers, 3x3 grid
    return df[cols].values.reshape(-1, 5, 3, 3)


def get_layer_sum(layer_3x3: np.ndarray) -> np.ndarray:
    """
    Returns the total energy sum of the 3x3 layer.

    Input shape: (N_events, 3, 3)
    Output shape: (N_events, 1)
    """
    return layer_3x3.sum(axis=(1, 2)).reshape(-1, 1)


def get_layer_max(layer_3x3: np.ndarray) -> np.ndarray:
    """
    Returns the maximum energy cell for a given 3x3 layer across all events.

    Input shape: (N_events, 3, 3)
    Output shape: (N_events, 1)
    """
    # Flatten the 3x3 grid to 9 elements and find the max
    return layer_3x3.reshape(layer_3x3.shape[0], -1).max(axis=1, keepdims=True)

def get_layer_neighbors_sum(layer_3x3: np.ndarray) -> np.ndarray:
    """
    Calculates the energy of all cells in the 3x3 grid excluding the max pixel.
    This effectively acts as the "neighbor sum" within the localized 3x3 space.

    Input shape: (N_events, 3, 3)
    Output shape: (N_events, 1)
    """
    total_sum = get_layer_sum(layer_3x3)
    max_val = get_layer_max(layer_3x3)
    return total_sum - max_val

def get_core_dominance(layer_3x3: np.ndarray) -> np.ndarray:
    """
    Feature 1: Max energy minus the sum of the other 8 cells.
    """
    max_val = get_layer_max(layer_3x3)
    neighbors_sum = get_layer_neighbors_sum(layer_3x3)

    return max_val - neighbors_sum

def get_dynamic_sparsity(layer_3x3: np.ndarray, threshold_multiplier=0.1) -> np.ndarray:
    """
    Feature 2: Count of cells with energy > (threshold * max_pixel).
    Uses multiplication by a constant to avoid division.
    """
    max_val = get_layer_max(layer_3x3)  # Shape: (N, 1)

    # Expand dims to (N, 1, 1) so we can broadcast against the (N, 3, 3) layer
    max_expanded = max_val.reshape(-1, 1, 1)

    # Multiplication by constant is allowed
    threshold = threshold_multiplier * max_expanded

    # Create boolean mask, convert to int, and sum the spatial 3x3 grid
    active_counts = (layer_3x3 > threshold).sum(axis=(1, 2))

    return active_counts.reshape(-1, 1).astype(np.float32)

def get_layer_features(df: pd.DataFrame, layer_index: int) -> np.ndarray:
    """
    Wrapper to extract the 3 key features for a specific layer.
    layer_index: 0=EM0, 1=EM1, 2=EM2, 3=EM3, 4=HAD
    """
    all_layers = get_5_layers_3x3(df)
    layer = all_layers[:, layer_index, :, :]

    dominance = get_core_dominance(layer)
    sparsity = get_dynamic_sparsity(layer)
    total_sum = get_layer_sum(layer)

    # Returns an (N, 3) matrix of dense features for this layer
    return np.hstack([dominance, sparsity, total_sum])

def extract_specific_feature(df: pd.DataFrame, layer_idx: int, feature_func) -> np.ndarray:
    """
    Helper function to dynamically extract a specific layer and apply a feature metric.
    layer_idx: 0=EM0, 1=EM1, 2=EM2, 3=EM3, 4=HAD
    """
    all_layers = get_5_layers_3x3(df)
    layer = all_layers[:, layer_idx, :, :]
    return feature_func(layer)


# =============================================================================
# NEW FEATURE ENGINEERING (EM2 12x12, Division Approximations, and EM2 3x3)
# =============================================================================

# --- 12x12 EM2 Features ---

def get_em2_max1(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts the maximum cell energy from the high-resolution 12x12 EM2 layer.
    """
    em2_3d = _get_em2_matrix(df)
    max_vals = em2_3d.reshape(em2_3d.shape[0], -1).max(axis=1)
    return max_vals.reshape(-1, 1)


def get_em2_max2(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts the second maximum cell energy from the 12x12 EM2 layer.
    Uses array partitioning to find the top 2 elements efficiently.
    """
    em2_3d = _get_em2_matrix(df)
    flat_em2 = em2_3d.reshape(em2_3d.shape[0], -1)

    # Isolate the top 2 values at the end of the array
    top2 = np.partition(flat_em2, -2, axis=1)[:, -2:]

    # The second maximum is the smaller of these two values
    max2_vals = top2.min(axis=1)
    return max2_vals.reshape(-1, 1).astype(np.float32)


def get_em2_maxdist(df: pd.DataFrame) -> np.ndarray:
    """
    Calculates the spatial Euclidean distance between the highest
    and second-highest energy cells in the 12x12 EM2 layer.
    """
    em2_3d = _get_em2_matrix(df)
    N = em2_3d.shape[0]
    flat_em2 = em2_3d.reshape(N, -1)

    # Get flat indices of the top 2 elements
    top2_idx = np.argpartition(flat_em2, -2, axis=1)[:, -2:]

    distances = np.zeros((N, 1), dtype=np.float32)

    for i in range(N):
        idx1, idx2 = top2_idx[i]

        # Convert flat indices back to 2D coordinates
        y1, x1 = np.unravel_index(idx1, (12, 12))
        y2, x2 = np.unravel_index(idx2, (12, 12))

        # squared Euclidean distance - (not allowed roots)
        distances[i, 0] = (y2 - y1) ** 2 + (x2 - x1) ** 2

    return distances


def get_em2_width(df: pd.DataFrame) -> np.ndarray:
    """
    Calculates the energy-weighted spatial variance (width) of the 12x12 EM2 shower.
    To comply with FPGA constraints, division by total energy is omitted.
    """
    em2_3d = _get_em2_matrix(df)
    N = em2_3d.shape[0]

    flat_em2 = em2_3d.reshape(N, -1)
    max_idx = flat_em2.argmax(axis=1)

    # Geometric center based on the maximum cell
    y_center, x_center = np.unravel_index(max_idx, (12, 12))

    widths = np.zeros((N, 1), dtype=np.float32)

    for i in range(N):
        y, x = np.ogrid[:12, :12]
        dist_sq = (y - y_center[i]) ** 2 + (x - x_center[i]) ** 2 # squared distances
        widths[i, 0] = np.sum(em2_3d[i] * dist_sq)

    return widths


# --- Division Approximations (FPGA Safe) ---

def get_frach_approx(df: pd.DataFrame) -> np.ndarray:
    """
    Approximates the hadronic energy fraction using subtraction instead of division.
    Calculated as: (HAD layer sum) - (Sum of EM0, EM1, EM2, EM3 layers).
    """
    all_layers = get_5_layers_3x3(df)
    layer_sums = all_layers.sum(axis=(2, 3))

    em_sum = layer_sums[:, 0:4].sum(axis=1)
    had_sum = layer_sums[:, 4]

    frach_approx = had_sum - em_sum
    return frach_approx.reshape(-1, 1).astype(np.float32)


def get_em2_maxratio_approx(df: pd.DataFrame) -> np.ndarray:
    """
    Approximates the relative size of the second peak (maxratio) for the 12x12 EM2
    using subtraction (max2 - max1) to avoid division.
    """
    max1 = get_em2_max1(df)
    max2 = get_em2_max2(df)
    ratio_approx = max2 - max1
    return ratio_approx.astype(np.float32)


# --- 3x3 EM2 Features ---

def get_em2_3x3_max2(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts the second maximum cell energy from the small 3x3 EM2 layer.
    """
    all_layers = get_5_layers_3x3(df)
    em2_3x3 = all_layers[:, 2, :, :]

    flat_em2 = em2_3x3.reshape(em2_3x3.shape[0], -1)
    top2 = np.partition(flat_em2, -2, axis=1)[:, -2:]

    max2_vals = top2.min(axis=1)
    return max2_vals.reshape(-1, 1).astype(np.float32)


def get_em2_3x3_maxratio_approx(df: pd.DataFrame) -> np.ndarray:
    """
    Approximates the maxratio for the 3x3 EM2 using subtraction (max2 - max1).
    """
    all_layers = get_5_layers_3x3(df)
    em2_3x3 = all_layers[:, 2, :, :]

    max1 = get_layer_max(em2_3x3)
    max2 = get_em2_3x3_max2(df)

    ratio_approx = max2 - max1
    return ratio_approx.astype(np.float32)


def get_em2_3x3_maxdist(df: pd.DataFrame) -> np.ndarray:
    """
    Calculates the spatial Euclidean distance between the highest
    and second-highest energy cells in the 3x3 EM2 layer.
    """
    all_layers = get_5_layers_3x3(df)
    em2_3x3 = all_layers[:, 2, :, :]
    N = em2_3x3.shape[0]

    flat_em2 = em2_3x3.reshape(N, -1)
    top2_idx = np.argpartition(flat_em2, -2, axis=1)[:, -2:]

    distances = np.zeros((N, 1), dtype=np.float32)

    for i in range(N):
        idx1, idx2 = top2_idx[i]

        y1, x1 = np.unravel_index(idx1, (3, 3))
        y2, x2 = np.unravel_index(idx2, (3, 3))

        distances[i, 0] = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) # TODO: whoops? should be without sqrt. Could this be why it's the best? need to fix

    return distances


def get_em2_top2_3x3_sqdist(df: pd.DataFrame) -> np.ndarray:
    """
    Calculates the squared Euclidean distance between the center of the highest
    sum 3x3 window and the second highest non-overlapping 3x3 window in the
    12x12 EM2 layer.
    """
    em2_3d = _get_em2_matrix(df)
    N = em2_3d.shape[0]

    # Get sums of all possible 3x3 windows (shape: N, 10, 10)
    window_sums = _all_3x3_window_sums(em2_3d)

    distances = np.zeros((N, 1), dtype=np.float32)

    for i in range(N):
        ws = window_sums[i].copy()

        # Find the 1st max 3x3 window
        flat_idx1 = np.argmax(ws)
        y1, x1 = np.unravel_index(flat_idx1, ws.shape)

        # Mask out overlapping windows to find a distinct 2nd peak
        y_min = max(0, y1 - 2)
        y_max = min(10, y1 + 3)
        x_min = max(0, x1 - 2)
        x_max = min(10, x1 + 3)
        ws[y_min:y_max, x_min:x_max] = -np.inf

        # Find the 2nd max 3x3 window
        flat_idx2 = np.argmax(ws)
        y2, x2 = np.unravel_index(flat_idx2, ws.shape)

        # Squared Euclidean distance between their positions
        # (Since window centers are just y+1, x+1, relative distance is identical)
        distances[i, 0] = (y2 - y1) ** 2 + (x2 - x1) ** 2

    return distances


def get_em2_6x6_maxdist(df: pd.DataFrame) -> np.ndarray:
    """
    Calculates the squared Euclidean distance between the highest and
    second-highest pixels on a lower-resolution 6x6 EM2 matrix.
    The 6x6 matrix is formed by summing 2x2 blocks of the original 12x12 matrix.
    """
    em2_3d = _get_em2_matrix(df)
    N = em2_3d.shape[0]

    # Efficiently downsample 12x12 to 6x6 by summing 2x2 non-overlapping blocks
    # Reshape splits the 12x12 into 6 groups of 2 for both rows and columns
    em2_6x6 = em2_3d.reshape(N, 6, 2, 6, 2).sum(axis=(2, 4))

    # Flatten the 6x6 to find the top 2 indices
    flat_em2_6x6 = em2_6x6.reshape(N, -1)

    # Get flat indices of the top 2 elements
    top2_idx = np.argpartition(flat_em2_6x6, -2, axis=1)[:, -2:]

    distances = np.zeros((N, 1), dtype=np.float32)

    for i in range(N):
        idx1, idx2 = top2_idx[i]

        # Convert flat indices back to 2D coordinates in the 6x6 grid
        y1, x1 = np.unravel_index(idx1, (6, 6))
        y2, x2 = np.unravel_index(idx2, (6, 6))

        # Squared Euclidean distance
        distances[i, 0] = (y2 - y1) ** 2 + (x2 - x1) ** 2

    return distances

FEATURE_REGISTRY = {
    # --- Original Features ---
    "core_physics": get_core_physics,
    "core_tensors": get_core_tensors,

    # --- High-Res EM2 Features (12x12) ---
    "em2_max": get_em2_max,
    "em2_max_neighbors_sum": get_em2_max_neighbors_sum,
    "em2_all_cells": get_all_em2_144,
    "em2_best_3x3_fraction": get_em2_best_3x3_fraction,
    "em2_outside_best_3x3_over_pt": get_em2_outside_best_3x3_over_pt,
    "em2_top3_3x3_features": get_em2_top3_3x3_features,

    # --- EM0 ---
    "em0_dominance": lambda df: extract_specific_feature(df, 0, get_core_dominance),
    "em0_sparsity": lambda df: extract_specific_feature(df, 0, get_dynamic_sparsity),
    "em0_sum": lambda df: extract_specific_feature(df, 0, get_layer_sum),

    # --- EM1 ---
    "em1_dominance": lambda df: extract_specific_feature(df, 1, get_core_dominance),
    "em1_sparsity": lambda df: extract_specific_feature(df, 1, get_dynamic_sparsity),
    "em1_sum": lambda df: extract_specific_feature(df, 1, get_layer_sum),

    # --- EM2 (3x3 Core) ---
    "em2_3x3_dominance": lambda df: extract_specific_feature(df, 2, get_core_dominance),
    "em2_3x3_sparsity": lambda df: extract_specific_feature(df, 2, get_dynamic_sparsity),
    "em2_3x3_sum": lambda df: extract_specific_feature(df, 2, get_layer_sum),

    # --- EM3 ---
    "em3_dominance": lambda df: extract_specific_feature(df, 3, get_core_dominance),
    "em3_sparsity": lambda df: extract_specific_feature(df, 3, get_dynamic_sparsity),
    "em3_sum": lambda df: extract_specific_feature(df, 3, get_layer_sum),

    # --- HAD ---
    "had_dominance": lambda df: extract_specific_feature(df, 4, get_core_dominance),
    "had_sparsity": lambda df: extract_specific_feature(df, 4, get_dynamic_sparsity),
    "had_sum": lambda df: extract_specific_feature(df, 4, get_layer_sum),

# --- Fresh EM2 Testing Features (12x12) ---
    "em2_max1": get_em2_max1,
    "em2_max2": get_em2_max2,
    "em2_maxdist": get_em2_maxdist,
    "em2_width": get_em2_width,

    # --- Approximation Features (FPGA Division Workarounds) ---
    "frach_approx": get_frach_approx,
    "em2_maxratio_approx": get_em2_maxratio_approx,

    # --- 3x3 EM2 Features (Hardware Constrained) ---
    "em2_3x3_max2": get_em2_3x3_max2,
    "em2_3x3_maxratio_approx": get_em2_3x3_maxratio_approx,
    "em2_3x3_maxdist": get_em2_3x3_maxdist,

    # --- trying more distance variants ---
    "em2_top2_3x3_sqdist": get_em2_top2_3x3_sqdist,
    "em2_6x6_maxdist": get_em2_6x6_maxdist,

    "tob_pt_only": _get_tob_pt,
}