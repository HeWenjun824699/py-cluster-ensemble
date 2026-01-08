import pandas as pd
import numpy as np
import os

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


def generate_brainnet_node(labels, consensus_matrix, save_path, split_output=True):
    """
    Generate a .node file for BrainNet Viewer.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels (N,), determines the Node Color.
    consensus_matrix : np.ndarray
        Consensus matrix (N, N), used to calculate Node Size.
        Size = Node Strength (Sum of consensus probabilities for the node).
    save_path : str
        Path to save the .node file.
    split_output : bool
        If True, automatically creates a subfolder and saves separate .node files for each cluster.
    """
    try:
        # Create directory if not exists
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(path)

        n_nodes = len(labels)

        # 1. Validate dimensions
        if consensus_matrix.shape[0] != n_nodes:
            print(
                f"[Error] Labels size ({n_nodes}) != Matrix size ({consensus_matrix.shape[0]}). Skip node generation.")
            return

        # 2. Get coordinates (X, Y, Z)
        coords = get_power_264_coords(n_nodes)

        # 3. Calculate Node Size based on Node Strength
        # Logic: Sum of the row in consensus matrix.
        # Higher sum -> More stable/central node in the consensus structure.
        node_sizes = np.sum(consensus_matrix, axis=1)

        # Optional: Normalize size if needed (e.g., Min-Max scaling)
        # currently keeping raw strength values.

        # 4. Process Labels (Color)
        # BrainNet Viewer requires positive integers (>=1) for color indices.
        labels_processed = labels.astype(int)
        if np.min(labels_processed) == 0:
            labels_processed += 1

        # 5. Generate File Content
        # Format: X  Y  Z  Color  Size  Label
        lines = []
        for i in range(n_nodes):
            x, y, z = coords[i]
            color = labels_processed[i]
            size = node_sizes[i]
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
