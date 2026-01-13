import os

import numpy as np
import pandas as pd
from nilearn import datasets


def get_power_264_coords(n_nodes):
    """
    Fetch Power 264 coordinates using nilearn.
    If nilearn is missing or node count mismatches, returns random or truncated coordinates.
    """
    coords = None
    try:
        power = datasets.fetch_coords_power_2011()
        coords = power.rois
        if isinstance(coords, pd.DataFrame):
            coords = coords.values
    except Exception as e:
        print(f"[Warning] Could not fetch coordinates: {e}")

    # 1. Fallback: Generate random coordinates if fetch failed
    if coords is None:
        return np.random.rand(n_nodes, 3) * 50

    # 2. Check dimensions: Ensure only 3 columns (X, Y, Z)
    if coords.shape[1] == 4:
        coords = coords[:, 1:4]

    # 3. Match node count
    if len(coords) != n_nodes:
        # If it is standard 264 but matrix is not, usually an error, but we handle it gracefully
        if n_nodes == 264 and len(coords) == 264:
            pass
        else:
            print(f"[Warning] Coordinate count ({len(coords)}) != Node count ({n_nodes}).")
            if len(coords) > n_nodes:
                coords = coords[:n_nodes]  # Truncate
            else:
                # Pad with random coordinates if not enough
                diff = n_nodes - len(coords)
                coords = np.vstack([coords, np.random.rand(diff, 3)])

    return coords


def get_cluster_to_name_mapping(predicted_labels, gt_dir):
    """
    Maps predicted cluster IDs to ground truth network names based on majority overlap.
    Logic ported from map.py.

    Args:
        predicted_labels: Array of predicted cluster IDs (from ICSC).
        gt_dir: Directory containing 'node_communities.npy' and 'label_names_map.npy'.

    Returns:
        dict: Mapping {predicted_cluster_id: 'Network_Name'}
    """
    gt_path = os.path.join(gt_dir, "node_communities.npy")
    names_path = os.path.join(gt_dir, "label_names_map.npy")

    # Check if ground truth files exist
    if not os.path.exists(gt_path) or not os.path.exists(names_path):
        print(f"[Warning] Ground truth files not found in {gt_dir}. Labels will be empty.")
        return {}

    try:
        # Load ground truth data
        # Ensure GT labels are integers
        gt_labels = np.load(gt_path).astype(int)
        label_names = np.load(names_path, allow_pickle=True)

        # Handle length mismatch (truncate to minimum length)
        min_len = min(len(predicted_labels), len(gt_labels))
        pred_trunc = predicted_labels[:min_len]
        gt_trunc = gt_labels[:min_len]

        mapping = {}
        unique_preds = np.unique(pred_trunc)

        for pid in unique_preds:
            # Find indices where prediction matches current cluster ID
            indices = np.where(pred_trunc == pid)[0]
            if len(indices) == 0: continue

            # Get corresponding ground truth values
            gt_values = gt_trunc[indices]

            # Majority vote: Find which GT label appears most in this predicted cluster
            counts = np.bincount(gt_values)
            best_gt_id = np.argmax(counts)

            # Map ID to Name (ensure index is within bounds)
            if best_gt_id < len(label_names):
                mapping[pid] = label_names[best_gt_id]
            else:
                mapping[pid] = f"Unknown_{pid}"

        return mapping

    except Exception as e:
        print(f"[Error] Failed to generate label mapping: {e}")
        return {}


def save_split_modules(lines, labels_processed, original_save_path):
    """
    Helper function: Splits the generated node lines into separate files per module.

    Parameters
    ----------
    lines : list
        The list of formatted string lines (content of the .node file).
    labels_processed : np.ndarray
        The adjusted labels (colors) used in the lines.
    original_save_path : str
        The path of the main .node file, used to determine output directory.
    """
    try:
        # 1. Determine Output Directory
        base_dir = os.path.dirname(original_save_path)
        filename = os.path.basename(original_save_path)
        file_root = os.path.splitext(filename)[0]

        output_dir = os.path.join(base_dir, f"{file_root}_modules")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 2. Identify Unique Modules
        unique_modules = np.unique(labels_processed)
        print(f"[NodeGen] Splitting into {len(unique_modules)} modules -> {output_dir}")

        # 3. Loop and Save
        for module_id in unique_modules:
            # Filter lines where the label matches module_id
            # We use enumerate to match list index with label array index
            module_lines = [line for i, line in enumerate(lines) if labels_processed[i] == module_id]

            if not module_lines:
                continue

            # Construct filename: e.g., Module_1.node
            out_name = os.path.join(output_dir, f"Module_{module_id}.node")

            with open(out_name, 'w') as f:
                f.write('\n'.join(module_lines))

    except Exception as e:
        print(f"[Warning] Failed to split modules: {e}")


def generate_brainnet_node(labels, consensus_matrix, data_path, save_path, split_output=True):
    """
    Generate a .node file for BrainNet Viewer.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels (N,), determines the Node Color.
    consensus_matrix : np.ndarray
        Consensus matrix (N, N), used to calculate Node Size.
        Size = Node Strength (Sum of consensus probabilities for the node).
    data_path: str
        Directory containing 'node_communities.npy' and 'label_names_map.npy'.
    save_path : str
        Path to save the .node file.
    split_output : bool
        If True, automatically creates a subfolder and saves separate .node files for each cluster.
    """
    try:
        # Create directory if not exists
        path = os.path.dirname(save_path)
        if not os.path.exists(path):
            os.makedirs(path)

        n_nodes = len(labels)

        # 1. Validate dimensions
        if consensus_matrix.shape[0] != n_nodes:
            print(
                f"[Error] Labels size ({n_nodes}) != Matrix size ({consensus_matrix.shape[0]}). Skip node generation.")
            return

        # 2. Get coordinates (X, Y, Z)
        coords = get_power_264_coords(n_nodes)

        # 3. Get Label Mapping
        label_map_dict = get_cluster_to_name_mapping(labels, data_path)

        # 4. Calculate Node Size based on Node Strength
        # Logic: Sum of the row in consensus matrix.
        # Higher sum -> More stable/central node in the consensus structure.
        node_sizes = np.sum(consensus_matrix, axis=1)

        # Optional: Normalize size if needed (e.g., Min-Max scaling)
        # currently keeping raw strength values.

        # 5. Process Labels (Color)
        # BrainNet Viewer requires positive integers (>=1) for color indices.
        labels_processed = labels.astype(int)
        if np.min(labels_processed) == 0:
            labels_processed += 1

        # 6. Generate File Content
        # Format: X  Y  Z  Color  Size  Label
        lines = []
        for i in range(n_nodes):
            x, y, z = coords[i]
            color = labels_processed[i]
            size = node_sizes[i]

            if label_map_dict:
                original_cluster_id = int(labels[i])
                raw_text = label_map_dict.get(original_cluster_id, '-')
                label_text = str(raw_text).replace(" ", "_")
            else:
                label_text = '-'  # Do not display text label

            # Use tab or space delimiter
            line = f"{x:.4f} {y:.4f} {z:.4f} {color} {size:.4f} {label_text}"
            lines.append(line)

        # 6. Write to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"[NodeGen] Successfully saved: {save_path}")

        # 7. Split Modules (Integrated Logic)
        if split_output:
            save_split_modules(lines, labels_processed, save_path)

    except Exception as e:
        print(f"[Error] Failed to generate node file: {e}")
