import os
import traceback
from typing import Optional

from pathlib import Path

# Import library components
from .. import io
from .. import generators
from .. import consensus
from .. import metrics


def consensus_batch(
        input_dir: str,
        output_dir: Optional[str] = None,
        save_format: str = "csv",
        consensus_method: str = 'cspa',
        generator_method: str = 'cdkmeans',
        nPartitions: int = 200,
        seed: int = 2026,
        maxiter: int = 100,
        replicates: int = 1,
        nBase: int = 20,
        nRepeat: int = 10,
        overwrite: bool = False
):
    """
    Execute the full cluster ensemble pipeline in batch mode for multiple datasets.

    This high-level API automates the entire workflow: it scans a directory for
    MATLAB (.mat) files, handles data loading, generates base partitions
    on-the-fly (if missing), executes consensus algorithms, performs performance
    evaluation, and persists results in various formats.

    Parameters
    ----------
    input_dir : str
        The directory path containing input .mat datasets. The function
        automatically scans for all .mat files within this folder.
    output_dir : str, optional
        The directory where experimental results will be saved. If None, it
        defaults to `input_dir` and creates subfolders named after the
        `consensus_method`.
    save_format : str, default="csv"
        The file format for saving evaluation results. Supports 'csv',
        'xlsx', and 'mat'.
    consensus_method : str, default='cspa'
        The name of the consensus algorithm to be executed (e.g., 'cspa',
        'mcla', 'hgpa', 'lwea', etc.).
    generator_method : str, default='cdkmeans'
        The generator algorithm to use if the input dataset contains only
        raw features (X). Options include 'cdkmeans', 'litekmeans', etc.
    nPartitions : int, default=200
        Total number of base partitions to generate per dataset. This is
        only utilized when the input file lacks pre-computed BPs.
    seed : int, default=2026
        Global random seed to ensure reproducibility of both base clustering
        generation and consensus algorithm execution.
    maxiter : int, default=100
        Maximum number of iterations allowed for the base clustering
        generation process.
    replicates : int, default=1
        The number of independent initializations/runs for the base
        clustering generator per partition.
    nBase : int, default=20
        The number of base partitions selected for each individual ensemble
        experiment (slice size from the total pool).
    nRepeat : int, default=10
        The number of independent experimental repetitions per dataset to
        facilitate statistical analysis of algorithm stability.
    overwrite : bool, default=False
        If True, existing output files will be overwritten. If False, the
        pipeline skips datasets that already have results in the output path.

    Returns
    -------
    None
    """

    # 1. Prepare directories
    input_path = Path(input_dir)
    # Check if output path is provided
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)

    if not output_path.exists():
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # 2. Get algorithm functions (Dynamically using getattr)
    # Convert generator_method, consensus_method to lowercase
    generator_method = generator_method.lower()
    consensus_method = consensus_method.lower()

    try:
        save_func = getattr(io, "save_results_" + save_format)
    except AttributeError:
        raise ValueError(f"Save format '{save_format}' not found in pce.io")

    try:
        consensus_func = getattr(consensus, consensus_method)
    except AttributeError:
        raise ValueError(f"Consensus method '{consensus_method}' not found in pce.consensus")

    try:
        generator_func = getattr(generators, generator_method)
    except AttributeError:
        raise ValueError(f"Generator method '{generator_method}' not found in pce.generators")

    # 3. Iterate over files
    mat_files = list(input_path.glob("*.mat"))
    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return

    print(f"\nFound {len(mat_files)} datasets. Starting batch process with [{consensus_method.upper()}]...")

    for file_path in mat_files:
        dataset_name = file_path.stem  # Get filename (without extension)

        # Check if output file exists
        save_path = output_path / f"{consensus_method.upper()}" / f"{dataset_name}_{consensus_method.upper()}.{save_format}"
        if not overwrite and save_path.exists():
            print(f"    - Skipping: {save_path} already exists.")
            continue

        # If not skipped, start processing
        print(f"\n>>> Processing: {dataset_name}")

        try:
            BPs = None
            Y = None

            # --- A & B. Data Loading and Probing (Core Modification) ---
            try:
                # Option 1: Attempt to load BPs and Y directly first
                # load_mat_BPs_Y will automatically handle 1-based indexing issues
                print(f"    - Attempting to load pre-computed BPs...")
                BPs, Y = io.load_mat_BPs_Y(file_path)
                print(f"    - Success: Pre-computed BPs found.")

            except IOError:
                # Option 2: If BPs not found (IOError), fallback to loading X and generating on the fly
                print(f"    - BPs not found. Fallback: Loading raw data (X)...")

                # If this also fails (e.g., file corrupted or no X/Y), IOError will be raised and caught by outer block
                X, Y = io.load_mat_X_Y(file_path)

                print(f"    - Generating BPs using {generator_method.upper()}...")

                # Run base clustering generator
                BPs = generator_func(X, Y, nPartitions=nPartitions, seed=seed, maxiter=maxiter, replicates=replicates)

            # --- C. Run Consensus ---
            print(f"    - Running Consensus: {consensus_method.upper()}...")
            labels, time_list = consensus_func(BPs, Y, nBase=nBase, nRepeat=nRepeat, seed=seed)

            # --- D. Evaluation ---
            print(f"    - Evaluating...")
            res = metrics.evaluation_batch(labels, Y, time_list)

            # --- E. Saving ---
            save_name = f"{dataset_name}_{consensus_method.upper()}.{save_format}"
            save_path = output_path / f"{consensus_method.upper()}" / save_name
            save_func(res, str(save_path))
            print(f"    - Saved to: {save_name}")

        except Exception as e:
            # Catch all exceptions (including load_mat_X_Y failure)
            print(f"!!! Error processing {dataset_name}: {e}")
            # Print stack trace for debugging
            # traceback.print_exc()

    print("\nBatch processing completed.")
