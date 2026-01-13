import os
import numpy as np
from multiprocessing import Pool
from .methods.ICSC.icsc_mul_core import ensure_dir
from .methods.ICSC.icsc_mul_core import single_multiple_run
from .methods.ICSC.icsc_sub_core import single_subject_run
from .methods.ICSC.utils import get_threshold


def icsc_mul_application(
        dataset,
        data_directory,
        save_dir,
        num_nodes: int = 264,
        num_threads: int = 1,
        num_runs: int = 1,
        max_labels: int = 21,
        min_labels: int = 5,
        percent_threshold: int = 100,
        heatmap_format: str = 'png'
):
    """
    Main controller function: Prepare data and start multiprocessing pool.

    Parameters
    ----------
    dataset : str
        Name of the dataset being processed.
    data_directory : str
        Path to the directory containing subject data.
    save_dir : str
        Directory where results will be saved.
    num_nodes : int, optional, default=264
        Number of nodes in the graph.
    num_threads : int, optional, default=1
        Number of threads (processes) to use for parallel execution.
    num_runs : int, optional, default=1
        Number of independent runs to execute.
    max_labels : int, optional, default=21
        Maximum number of clusters/labels.
    min_labels : int, optional, default=5
        Minimum number of clusters/labels.
    percent_threshold : int, optional, default=100
        Threshold percentage for graph connectivity.
    heatmap_format : str, optional, default='png'
        Format to save the consensus heatmap ('png' or 'pdf').

    Returns
    -------
    None
        This function does not return a value. It prints the status of each run.
    """
    ensure_dir(save_dir)

    # Get list of subject directories
    sub_list = []
    if os.path.isdir(data_directory):
        file_list = os.listdir(data_directory)
        for file in file_list:
            if file.startswith('subject'):
                sub_list.append(file)
    else:
        print(f"Error: Data directory not found: {data_directory}")
        return

    print(f"\nFound {len(sub_list)} subjects. Starting {num_runs} runs with {num_threads} threads...")

    # Prepare parameter list
    params = []
    for run in range(num_runs):
        # Parameter order must match single_run unpacking order
        p = (run, data_directory, percent_threshold, sub_list,
             max_labels, min_labels, num_nodes, dataset, save_dir, heatmap_format)
        params.append(p)

    # Start multiprocessing
    if num_threads > 1:
        with Pool(num_threads) as p:
            for val in p.map(single_multiple_run, params):
                print(f'Completed Run: {val}')
    else:
        # Single-thread debug mode
        for param in params:
            val = single_multiple_run(param)
            print(f'Completed Run: {val}')


def icsc_sub_application(
        dataset,
        data_directory,
        save_dir,
        num_nodes: int = 264,
        num_threads: int = 1,
        max_labels: int = 21,
        min_labels: int = 5,
        percent_threshold: int = 100,
        heatmap_format: str = 'png'
):
    """
    Subject-Level Main Controller.

    1. Iterates through all Subject directories.
    2. Loads Session data (.npy) for each Subject.
    3. Performs preprocessing (Thresholding).
    4. Assembles parameters and starts the multiprocessing pool (one process per Subject).

    Parameters
    ----------
    dataset : str
        Name of the dataset being processed.
    data_directory : str
        Path to the directory containing subject data.
    save_dir : str
        Directory where results will be saved.
    num_nodes : int, optional, default=264
        Number of nodes in the graph.
    num_threads : int, optional, default=1
        Number of threads (processes) to use for parallel execution.
    max_labels : int, optional, default=21
        Maximum number of clusters/labels.
    min_labels : int, optional, default=5
        Minimum number of clusters/labels.
    percent_threshold : int, optional, default=100
        Threshold percentage for graph connectivity.
    heatmap_format : str, optional, default='png'
        Format to save the consensus heatmap ('png' or 'pdf').

    Returns
    -------
    None
        This function does not return a value. It prints the status of each subject run.
    """
    ensure_dir(save_dir)

    # Get list of subject directories
    sub_list = []
    if os.path.isdir(data_directory):
        file_list = os.listdir(data_directory)
        for file in file_list:
            if file.startswith('subject'):
                sub_list.append(file)
    else:
        print(f"Error: Data directory not found: {data_directory}")
        return

    print(f"\nFound {len(sub_list)} subjects. Preparing data and starting pool with {num_threads} threads...")

    params = []

    # Iterate through each subject (In Subject Level, a "Run" corresponds to processing one Subject)
    for run_id, subject in enumerate(sub_list):
        subject_session_dir = os.path.join(data_directory, subject)

        # Initialize data containers
        subject_session_list = []  # Store Session ID (0, 1, 2...)
        subject_session_data = dict()  # Store actual matrix data
        idx = 0

        # Iterate through all Session files in the subject directory
        if os.path.isdir(subject_session_dir):
            for subject_session_file in os.listdir(subject_session_dir):
                # According to legacy logic, only process files ending with _corr.npy
                if subject_session_file.endswith('_corr.npy'):
                    file_path = os.path.join(subject_session_dir, subject_session_file)
                    try:
                        # 1. Load data
                        session = np.load(file_path)

                        # 2. Preprocessing: Thresholding
                        # Note: Ensure utils.get_threshold is available
                        threshold_val = get_threshold(session, percent_threshold)
                        session = session * (session > threshold_val)

                        # 3. Preprocessing: Set diagonal to zero
                        np.fill_diagonal(session, 0)

                        # 4. Store in dictionary
                        subject_session_data[idx] = session
                        subject_session_list.append(idx)
                        idx += 1
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        # Add to task list only if the Subject has valid data
        if len(subject_session_list) > 0:
            # Parameter order must strictly match single_subject_run unpacking order
            # (run_id, percent_threshold, session_ids, sessions_data, max, min, nodes, dataset, save_dir)
            p = (run_id, percent_threshold, subject_session_list,
                 subject_session_data, max_labels, min_labels,
                 num_nodes, dataset, save_dir, heatmap_format,
                 subject_session_dir)
            params.append(p)
        else:
            print(f"Warning: No valid session data found for {subject}")

    print(f"Data preparation complete. Processing {len(params)} subjects...")

    # Start multiprocessing
    if num_threads > 1:
        with Pool(num_threads) as p:
            for val in p.map(single_subject_run, params):
                print(f'Completed Subject Run ID: {val}')
    else:
        # Single-thread debug mode
        for param in params:
            val = single_subject_run(param)
            print(f'Completed Subject Run ID: {val}')
